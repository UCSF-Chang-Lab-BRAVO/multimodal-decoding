from train.ctc_decoding import greedy_beam_wer_cer
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
import numpy as np
import torchaudio
def test_ensemble(model, test_loader, greedy, beam_search_decoder,
                  texts, tokens, wandb_name ='final_te_', device='cuda',
                  print_greedy=True, verbose=True, printall=False,
                  print_hypo=False,
                  realtime_eval=False):
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
#     model.eval()
    loss_fn = nn.CTCLoss()
    all_emish = []
    wers = []
    pers = []
    cers = []
    with torch.no_grad():
        total_loss, total_samps = 0, 0
        total_wer = 0
        total_cer=0
        total_gcer = 0
        total_per = 0
        gts, transcripts = [], []
        for x, y, l, targ_len, gtsent, _ in test_loader: 
            x = x.float().to(device)
            y = y.long().to(device)
            l = l.int().to(device)
            if not texts is None:
                gtsent = gtsent.long().cpu().numpy()
            else: 
                gtsent = gtsent
            targ_len = targ_len.int().to(device)
            models = [model]
            for k, model in enumerate(models): 
                model.eval()
                model.to(device)
                if k == 0: 
                    emissions, _,  lengths = model(x, l)
                    emissions = F.softmax(emissions, dim=-1)
            
            all_emish.append(emissions.detach().cpu().numpy())
            emissions = torch.log(emissions)
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
                    wers.append(wer)
                    pers.append(per)
                    cers.append(cer)
                    total_wer += wer
                    total_cer += cer
                    total_per += per
                    total_gcer += greedy_cer
                   
                
        if print_greedy and verbose:
            print('net wer', total_wer/total_samps)
            print('net cer', total_cer/total_samps)
            print('greedy per', total_gcer/total_samps)
            print('beam per', total_per/total_samps)
            
        wandb.log({
            wandb_name + 'wer': total_wer/total_samps,
            wandb_name + 'med_wer':  np.median(wers),
            wandb_name + 'per': total_per/total_samps,
            wandb_name + 'med_per': np.median(pers),
            wandb_name + 'cer' : total_cer/total_samps,
            wandb_name + 'med_cer' : np.median(cers),
    
        })
        
        if realtime_eval:
#             print('wers:', wers, len(wers))
#             print('pers:', pers)
#             print('cers:', cers)
            wandb.log({
                'allrt_wer':wers,
                'allrt_cer':cers, 
                'allrt_per':pers, 
                'gts':gts, 
                'trans':transcripts
            })
    
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
            return epoch_loss, model, total_wer/total_samps, total_cer/total_samps, total_per/total_samps, wers, pers, cers,  transcripts, gts, all_emish
        else:
            return epoch_loss, model, None, None, None, wers, pers, cers, transcripts, gts, all_emish