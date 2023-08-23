"""
This file includes functions needed for decoding samples. This includes synthesizing audio, loading models and parameters
and applying scalers
"""
import collections
import os

import joblib
import neural_synthesis
import torch
import yaml
from tqdm import tqdm


def load_yaml_config(root_dir, experiment_name, run_name, local=True):
    """load yamel config file specified by run_name """
    if local:
        config_path = os.path.join(root_dir, 'torch_models', experiment_name, run_name, 'config.yaml')
    else:
        config_path = os.path.join(root_dir, experiment_name, run_name, 'config.yaml')
    with open(config_path, 'r') as yaml_config:
        config = yaml.load(yaml_config, Loader=yaml.Loader)
    return config


def load_model_from_yaml(root_dir, experiment_name, run_name, checkpoint_number, device='cpu', return_config=False,
                         local=True):
    """ loads the model """

    config = load_yaml_config(root_dir, experiment_name, run_name, local=local)
    if local:
        checkpoint_path = os.path.join(root_dir, 'torch_models', experiment_name, run_name,
                                       'model_' + str(checkpoint_number) + '.pt')
    else:
        checkpoint_path = os.path.join(root_dir, experiment_name, run_name,
                                       'model_' + str(checkpoint_number) + '.pt')

    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "CnnRnnClassifier"),
    )
    model = model_class(**config["model_params"]).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    if return_config:
        return model, config
    else:
        return model


def load_model_ecog2x_to_gimdecode(model_dir, model_name, checkpoint_number, device='cpu',
                                   return_config=False, generative=False):
    """ loads the model """
    config_path = os.path.join(model_dir, model_name, 'config.yaml')
    with open(config_path, 'r') as yaml_config:
        config = yaml.load(yaml_config, Loader=yaml.Loader)
    checkpoint_path = os.path.join(model_dir, model_name, 'model_' + str(checkpoint_number) + '.pt')

    model_class = getattr(
        neural_synthesis.models,
        config.get("model_type", "CnnRnnClassifier"),
    )
    model = model_class(**config["model_params"]).to(device)

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    if return_config:
        return model, config
    else:
        return model


def vocode_intermediates(Y, Y_, vocoder, vocoder_config, device='cpu', num_examples=None):
    """ Vocode intermediate representations """
    vocoder.to(device)
    yw_ = []  # predictions using ground truth intermediates
    w_ = []  # predictions using decoded outputs
    vocoder_inputs = []
    for output in vocoder_config["dataloader_params"]['input_types']:
        vocoder_inputs.append(output[:3])
    for i, ((output_type, y), (_, y_)) in enumerate(zip(Y.items(), Y_.items())):
        if output_type[:3] in vocoder_inputs:
            if num_examples is None:
                n_samples = y.shape[0]
            else:
                n_samples = num_examples
            for n in tqdm(range(n_samples)):
                yw_.append(vocoder(torch.unsqueeze(y[n, :, :].to(device), 0)))
                w_.append(vocoder(torch.unsqueeze(y_[n, :, :].to(device), 0)))
    w_ = torch.cat(w_, dim=0)
    yw_ = torch.cat(yw_, dim=0)
    return w_, yw_


def load_scalers(data_names, subject, root_dir):
    """Loads sklearn scalers (e.g. z-score) for input and output data"""
    scalers = collections.defaultdict(bool)
    for data_name in data_names:
        try:
            scalers[data_name] = joblib.load(
                os.path.join(root_dir, 'scalers', subject
                             + '_' + data_name + '.pkl'))
        except FileNotFoundError:
            print('Scaler for %s not found.' % data_name)
    return scalers


def _apply_scalers(data_names, scalers, data, invert=False):
    """Applies scalers to data"""
    scaled_data = []
    for data_name, x in zip(data_names, data):
        if scalers[data_name]:
            if invert:
                scaled_data.append(scalers[data_name].inverse_transform(x))
            else:
                scaled_data.append(scalers[data_name].transform(x))
        else:
            scaled_data.append(x)
    return tuple(scaled_data)


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = "".join([self.labels[i] for i in indices])
        return joined.replace("|", " ").strip().split()
