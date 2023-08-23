""" Script which trains the models """
import argparse
import os
from collections import defaultdict

import torch
import torch.nn.functional as F
import wandb
import yaml
from tqdm import tqdm

import neural_synthesis.models
from neural_synthesis import ctc_utils
from neural_synthesis import utils
from neural_synthesis.data.dataloaders import get_dataloaders

os.environ["WANDB_SILENT"] = "true"


def train_model(config, device, train_dataloader, test_dataloader):
    """Train the model"""

    # define models, criterion, scheduler, and optimizer
    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "CnnRnnClassifier"),
    )
    model = {
        "model": model_class(
            **config["model_params"],
        ).to(device)
    }
    optimizer_class = getattr(
        neural_synthesis.optimizers,
        config.get("model_optimizer_type", "RAdam"),
    )
    optimizer = {
        "model": optimizer_class(
            model["model"].parameters(),
            **config["model_optimizer_params"],
        )
    }
    scheduler_class = getattr(
        torch.optim.lr_scheduler,
        config.get("model_scheduler_type", "StepLR"),
    )
    scheduler = {
        "model": scheduler_class(
            optimizer=optimizer["model"],
            **config["model_scheduler_params"],
        )
    }
    criterion = {}
    if (config['use_ctc_loss'], False):
        criterion['ctc'] = F.ctc_loss
    else:
        config['use_ctc_loss'] = False
    print('Created model.')

    # checkpointing and config save
    checkpoint_dir = os.path.join(config['root_dir'], 'torch_models', config['experiment_name'], config['run_name'])
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    with open(os.path.join(checkpoint_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, Dumper=yaml.Dumper)
    checkpointer = utils.Checkpointer(checkpoint_dir)

    # train the model
    print('Training model.')
    global_step = 0
    local_train_step = 0
    total_train_loss = defaultdict(float)
    total_eval_loss = defaultdict(float)
    eval_metrics = defaultdict(float)

    for epoch in range(config['epochs']):
        for batch in tqdm(train_dataloader):

            #######################
            #         IO          #
            #######################
            x = utils.repackage_data(batch, config['dataloader_params']['input_types'], device, noise=config['chance'])
            y = utils.repackage_data(batch, config['dataloader_params']['output_types'], device)
            x = utils.tuple_to_obj(x)
            y = utils.tuple_to_obj(y)

            #######################
            #      Inference      #
            #######################

            # Run forward pass
            loss = 0.0
            if len(x.shape) == 2:
                x = torch.unsqueeze(x, 2)
            print("Input shape: ", x.shape)
            y_ = model["model"](x)

            # Compute loss
            if config['use_ctc_loss']:

                # get sequences
                sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
                target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
                blank = config['model_params']['n_classes'] - 1
                estimates = F.log_softmax(y_, dim=-1).permute((1, 0, 2))
                input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0],
                                           dtype=torch.int64).to(device)

                # compute ctc loss
                ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                            zero_infinity=True) * config['ctc_params']['loss_lambda'][0]
                loss += ctc_loss
                total_train_loss["train/pho_ctc_loss"] += ctc_loss.item()

            total_train_loss["train/loss"] += loss

            # Backpropogate and optimize
            optimizer["model"].zero_grad()
            loss.backward()
            if config["model_grad_norm"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    model["model"].parameters(),
                    config["model_grad_norm"],
                )
            optimizer["model"].step()
            if config["model_scheduler_type"] == "ReduceLROnPlateau":
                scheduler["model"].step(loss)
            else:
                scheduler["model"].step()
            global_step += 1
            local_train_step += 1
            print("STEP: ", global_step, " COMPLETE")

            #######################
            #      Evaluation     #
            #######################
            # evaluate and save model
            if global_step % config['steps_per_summary'] == 0:
                model['model'].eval()
                local_eval_step = eval_model(config, device, model, global_step, criterion, test_dataloader,
                                             total_eval_loss,
                                             eval_metrics)
                model['model'].train()
                if config['use_wandb']:
                    for loss_name, loss_val in total_train_loss.items():
                        wandb.log({loss_name: loss_val}, step=global_step, commit=False)
                    for loss_name, loss_val in total_eval_loss.items():
                        wandb.log({loss_name: loss_val}, step=global_step, commit=False)
                    for metric_name, metric_val in eval_metrics.items():
                        wandb.log({metric_name: metric_val.mean()}, step=global_step, commit=False)
                total_train_loss = defaultdict(float)
                total_eval_loss = defaultdict(float)
                eval_metrics = defaultdict(float)
                torch.save(model['model'].state_dict(), checkpointer(global_step))
                local_train_step = 0


def eval_model(config, device, model, global_step, criterion, test_dataloader, total_eval_loss, eval_metrics,
               encoder_config=None):
    for eval_steps_per_epoch, test_batch in enumerate(tqdm(test_dataloader), 1):

        # eval one step
        eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics)

        if eval_steps_per_epoch > config['steps_within_evaluation']:
            break
    return eval_steps_per_epoch


def eval_step(config, criterion, device, model, total_eval_loss, test_batch, eval_metrics):
    """Evaluate model one step."""

    # get data and run inference
    x = utils.repackage_data(test_batch, config['dataloader_params']['input_types'], device, noise=config['chance'])
    y = utils.repackage_data(test_batch, config['dataloader_params']['output_types'], device)
    x = utils.tuple_to_obj(x)
    y = utils.tuple_to_obj(y)
    loss = 0.0
    if len(x.shape) == 2:
        x = torch.unsqueeze(x, 2)
    y_ = model["model"](x)

    # save losses
    loss = 0.0
    if config['use_ctc_loss']:

        # get sequences
        sequences = y.to(device=device, dtype=torch.int64)  # recall discrete units are 1-D so shape B x DU
        target_lengths = torch.full((sequences.shape[0],), sequences.shape[1]).to(device, dtype=torch.int64)
        blank = config['model_params']['n_classes'] - 1
        estimates = F.log_softmax(y_, dim=-1).permute((1, 0, 2))
        input_lengths = torch.full(size=(estimates.shape[1],), fill_value=estimates.shape[0], dtype=torch.int64).to(
            device)
        ctc_loss = criterion['ctc'](estimates, sequences, input_lengths, target_lengths, blank=blank,
                                    zero_infinity=True) * config['ctc_params']['loss_lambda'][0]

        # compute ctc loss
        loss += ctc_loss
        total_eval_loss["test/pho_ctc_loss"] += ctc_loss.item()
        ctc_decoder = ctc_utils.Decoder(blank_index=blank, silent=[None], remove_rep=True)
        estimated_sequences = torch.argmax(estimates, dim=-1)
        cer = ctc_decoder.phone_word_error(estimated_sequences.T, sequences)
        total_eval_loss["test/cer"] += cer
    total_eval_loss["test/loss"] += loss

if __name__ == "__main__":

    # parse arguments provided
    parser = argparse.ArgumentParser(
        description="Supervised ECoG-to-X training"
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="name of model to be used",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="experiment name",
    )
    parser.add_argument(
        "--run_name",
        type=str,
        required=True,
        help="run name",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="yaml format configuration file. only the name needed",
    )
    parser.add_argument(
        "--train_filename",
        type=str,
        required=True,
        help=".txt file used for training set. only the name needed",
    )
    parser.add_argument(
        "--test_filename",
        type=str,
        required=True,
        help=".txt file used for eval set. only the name needed",
    )
    parser.add_argument(
        "--root_dir",
        type=str,
        required=True,
        help="gimlet_data directory (e.g. userdata/username/gimlet_data)",
    )
    parser.add_argument(
        "--debug",
        type=bool,
        default=False,
        help="Set to True to put in debug mode, which cycles through evaluation faster.",
    )
    parser.add_argument(
        "--chance",
        type=bool,
        default=False,
        help="Set to True to train a model on noise inputs",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default='bravo3',
        help="subject",
    )
    parser.add_argument(
        "--train_data_fraction",
        type=float,
        default=1.0,
        help="Data fraction for training set",
    )
    args = parser.parse_args()

    # choose device
    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # load config file
    with open(utils.get_config_file(args.config)) as f:
        experiment_config = yaml.load(f, Loader=yaml.Loader)
    with open(utils.get_config_file("default_config.yaml")) as f:
        default_config = yaml.load(f, Loader=yaml.Loader)
    config = utils.merge_yaml_configs(experiment_config, default_config)
    if args.train_data_fraction < 1.0:
        config["dataloader_params"]["train_data_fraction"] = args.train_data_fraction
    config.update(vars(args))

    if config['debug']:
        config['steps_within_evaluation'] = 5
        config['steps_per_summary'] = 5
        config['outdir'] = 'debug'
        config['num_save_intermediate_results'] = 4
        device = torch.device('cpu')

    # initialize wandb
    if config['use_wandb']:
        wandb.init(project=args.experiment_name, config=config, name=args.run_name)

    # load and build dataset
    print(config["dataloader_params"]['input_types'])
    print(config['train_filename'])
    print(config['test_filename'])
    print(config['subject'])
    train_dataloader, test_dataloader = get_dataloaders(config['train_filename'], config['test_filename'],
                                                        **config["dataloader_params"], root_dir=config['root_dir'],
                                                        config=config)


    # train the model
    train_model(config, device, train_dataloader, test_dataloader)
