from torch.nn import CTCLoss
import torch
import torch.nn.functional as F
import torch.nn as nn
import copy
from .ctc_decoding import greedy_beam_wer_cer
import numpy as np
import torchaudio

# import wandb

def train_f(model, train_loader, optimizer, device): 
    """
    Train a ctc model. 
    """
    total_loss = 0
    total_samps = 0
    model.train()
    loss_fn = nn.CTCLoss()
    for x, y, l, targ_len, _ in train_loader: 
        x = x.float().to(device)
        y = y.long().to(device)
        l = l.int().cpu()
        targ_len = targ_len.int().cpu()
#         print('in loop', x.shape)
        emissions, lengths = model(x, l)
        emissions = F.log_softmax(emissions, dim=-1)
        
        optimizer.zero_grad()
#         print(emissions.shape, y.shape)
        loss = loss_fn(emissions, y, lengths, targ_len)
        total_loss += loss.item()
        total_samps += x.shape[0]
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1)
        optimizer.step()
        
    epoch_loss = total_loss/total_samps
    return epoch_loss, model

def test_f(model, test_loader, optimizer, device, greedy, beam_search_decoder, texts, tokens, print_greedy=False, verbose=True, printall=False,
          print_hypo=False, text_direct=False):
    """
    
    Tests model loss,
        if the print_greedy flag is true, then it calculates wer/cer/per
    
    returns updated model, loss, and metrics, if they were evaluated, 
        otherwsie none. 
    
    Inputs: 
    Greedy =greedy ctc deocder
    beam_search_decoder = torchaudio ctc decoder
    print_greedy = True  - 
        Then wer, cer, per are evaluated
    verbose = True - 
        Then we print out predictions to get a flavor of what was going on :D
    texts - the ground truth text. May not need for audio,m then put in None.
    
    """
    model.eval()
    loss_fn = nn.CTCLoss()
    with torch.no_grad():
        total_loss, total_samps = 0, 0
        total_wer = 0
        total_cer=0
        total_gcer = 0
        total_per = 0
        gts, transcripts = [], []
        for x, y, l, targ_len, gtsent in test_loader: 
            x = x.float().to(device)
            y = y.long().to(device)
            l = l.int().to(device)
            if not texts is None:
                gtsent = gtsent.long().cpu().numpy()
            else: 
                gtsent = gtsent
            targ_len = targ_len.int().to(device)

            emissions, lengths = model(x, l)
            
            emissions = F.log_softmax(emissions, dim=-1)
            loss = loss_fn(emissions, y, lengths, targ_len)
            total_loss += loss.item()
            total_samps += x.shape[0]
      
            
            ### The code to look at the text.  We dont want to do this every trial since
            # it can be expensive.
            if print_greedy:
                 for k , e in enumerate(emissions.permute(1,0, 2)):
                    gt_phones = [tokens[yy] for yy in y.detach().cpu().numpy()[k] if not yy == -1 and not yy == 0]
                    if not texts is None:
                        gt  = texts[int(gtsent[k])]
                    else: 
                        gt = gtsent[k]
                        # Get the ground truth text. 
                    greedy_cer, gt_, transcript,cer,  wer, per = greedy_beam_wer_cer(e, gt, 
                                                                                     gt_phones, 
                                                                                     greedy,
                                                                                     beam_search_decoder,
                                                                                    print_hypo)
                    gts.append(gt_) 
                    transcripts.append(transcript)
                    total_wer += wer
                    total_cer += cer
                    total_per += per
                    total_gcer += greedy_cer
                     
        if print_greedy and verbose:
            print('net wer', total_wer/total_samps)
            print('net cer', total_cer/total_samps)
            print('greedy per', total_gcer/total_samps)
            print('beam per', total_per/total_samps)
    
        ctr =0 
        randomsent = np.arange(len(gts))
        np.random.shuffle(randomsent)
        for k in randomsent:
            gt = gts[k]
            trans = transcripts[k]
            wer = (torchaudio.functional.edit_distance(gt.split(' '), trans.split(' '))/len(gt.split(' ')))
            cer = (torchaudio.functional.edit_distance(gt, trans)/len(gt))
            if verbose: 
                print('gt', gt) 
                print('t', trans, 'wer: %.2f' %wer, 'cer: %.2f' %cer)
            ctr+=1
            if ctr > 10 and not printall: 
                break   
        epoch_loss = total_loss/total_samps
        # Currently I dont eval wer/cer every time because it is a bit computationally expesnsive.
        if print_greedy:
            return epoch_loss, model, total_wer/total_samps, total_cer/total_samps, total_per/total_samps
        else:
            return epoch_loss, model, None, None, None
    
    
def train_loop(model, train_loader, test_loader, optimizer, device, 
               texts, greedy, beam_search_decoder, tokens, 
               patience=10, start_eval=50, 
              wandb_log=False, max_epochs=1000 ,wercalcrate=3,
              checkpoint_dir=None, train=True, printall=False, 
              print_hypo=False, text_direct=False):
    """
    Train (and evaluate) the CTC model
    
    Non self-explanatory inputs: 
    Device = cuda or cpu
    texts = list of the ground truth text
    greedy= greedyCTCDecoder
    beamsearchdecoder = torch ctc deocder
    patience = how long to wait for training to improve. We will do patience based on wer could be made adjustable 
    start_eval - what epoch do we want to start evlauting WER/PER. 
    wandb_log - log results on wandb. 
    wercalcrate  - how often to calculate the wer
    """
    
    best_wer = np.inf
    best_model = None
    patience_ctr = 0
    
    for epoch in range(max_epochs):
        if train:
            tr_loss, model = train_f(model, train_loader, optimizer, device)
        else: 
            tr_loss = np.nan
        te_loss, model, wer, cer, per = test_f(model, test_loader, optimizer, 
                                               device, greedy,  beam_search_decoder, texts, tokens,
                                               print_greedy=(epoch%wercalcrate==0 and epoch >= start_eval), printall=printall, 
                                               print_hypo=print_hypo, text_direct=text_direct)
        print('epoch', epoch, 'tr loss: %.3f' %tr_loss, 'te_loss: %.3f' %te_loss, flush=True)
        if not wer is None: 
            if wandb_log:
                import wandb
                wandb.log({
                    'tr_loss':tr_loss,
                    'te_loss':te_loss,
                    'wer':wer,
                    'cer':cer,
                    'per':per, 
                    'patience_ctr':patience_ctr,
                    'best_wer':min(1, best_wer)
                })
            if wer < best_wer: 
                best_wer = wer
                patience_ctr = 0
                best_model = copy.deepcopy(model)
                if not checkpoint_dir is None:
                    if wer < .85:
                        import os
                        torch.save(model.state_dict(), os.path.join(checkpoint_dir, str(wandb.run.name) + f".pth"))
            else:
                patience_ctr +=1
                
        if patience_ctr > patience:
            break
                
    return best_model