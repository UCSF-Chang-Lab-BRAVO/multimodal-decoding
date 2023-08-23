import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.functional import edit_distance

def predict_greedy(model, test_loader, device='cuda', max_length=100, enc_for_s2s=None, gt_text=None, gt_ph=None):
    """Make predictions for the given inputs using greedy inference.

    Args:
    model: A sequence-to-sequence model.
    sentences: A list of input sentences, represented as strings.
    max_length: The maximum length at which to truncate outputs in order to
      avoid non-terminating inference.

    Returns:
    A list of predicted translations, represented as strings.
    """

    # Requirement: your implementation must be batched. This means that you should
    # make only one call to model.encode() at the start of the function, and make
    # only one call to model.decode() per inference step.
    rev_enc = {v:k for k, v in enc_for_s2s.items()}
    bos_id = enc_for_s2s['<sos>']
    eos_id = enc_for_s2s['<eos>']
    sentences = []
    gts = []
    gtp = []
    with torch.no_grad():
        for x, l, t, inds in test_loader: 
            x = x.float().to(device)
            l = l.long().to('cpu')
            t = t.long().to(device)
            encoder_output, encoder_mask, encoder_hidden = model.encode(x, l)
            decoded_output = torch.zeros((max_length+1, x.shape[0])).long()
            decoded_output[0, :] = bos_id # set first thing to beginning of sent.
            for step in range(max_length):
                if step == 0: 
                    dec_hidden = encoder_hidden
                logits, dec_hidden, _ = model.decode(torch.unsqueeze(decoded_output[step], dim=0), dec_hidden,
                                                    encoder_output, encoder_mask)
                # Logit shape = L, BS, V
                preds = torch.argmax(logits.cpu(), dim=-1)
                decoded_output[step+1] = preds.long()

            # Implementation tip: once an EOS token has been generated, force the output
            # for that example to be padding tokens in all subsequent time steps by
            # adding a large positive number like 1e9 to the appropriate logits.
      
            for k, decoded_sent in enumerate(decoded_output.cpu().T): 
                string_form = []
                for d in decoded_sent[1:]:
                    d = d.item()
                    string_form.append(d)
                    if d == eos_id: 
                        break
                sentences.append([rev_enc[k] for k in string_form])
                gts.append(gt_text[inds[k]])
                gtp.append(gt_ph[inds[k]])
        return sentences, gts, gtp

    
def calc_greedy_per(model, test_loader, device, enc_for_s2s, gt_text, labels):
    decodes, gts, gtp = predict_greedy(model, test_loader, device, max_length=100, enc_for_s2s=enc_for_s2s, gt_text=gt_text, gt_ph =labels['ph_label'].values)


    
    net_per = []
    for d, g in zip(decodes, gtp):
        per = edit_distance(d[:-1], g)/len(g)
        net_per.append(per)
    return np.mean(per)



def predict_beam_s2s(model, test_loader,device='cuda', k=5, max_length=45, gt_text=None, gt_ph=None, enc_for_s2s=None):
    """Make predictions for the given inputs using beam search.

    Args:
    model: A sequence-to-sequence model.
    sentences: A list of input sentences, represented as strings.
    k: The size of the beam.
    max_length: The maximum length at which to truncate outputs in order to
      avoid non-terminating inference.

    Returns:
    A list of beam predictions. Each element in the list should be a list of k
    strings corresponding to the top k predictions for the corresponding input,
    sorted in descending order by score.
    """

    # Requirement: your implementation must be batched. This means that you should
    # make only one call to model.encode() at the start of the function, and make
    # only one call to model.decode() per inference step.
    rev_enc = {v:k for k, v in enc_for_s2s.items()}
    # YOUR CODE HERE
    bos_id = enc_for_s2s['<sos>']
    eos_id = enc_for_s2s['<eos>']
    pad_id = enc_for_s2s['-']
    print(bos_id)
    # only one call to model.decode() per inference step.
    with torch.no_grad():
        sentences = []
        gts, gtp = [], []
        for x, l, t, inds in test_loader: 
            x = x.float().to(device)
            l = l.long().to('cpu')
            t = t.long().to(device)
            encoder_output, encoder_mask, encoder_hidden = model.encode(x, l)

            decoded_output = torch.zeros((max_length+1, x.shape[0])).long() # LEN +1, BS
            decoded_output[0, :] = bos_id
            # set first thing to beginning of sent.
            for step in range(max_length):
    #             print('step', step)
                if step == 0: 
                    dec_hidden = encoder_hidden

#                 import ipdb
#                 ipdb.set_trace()

                logits, dec_hidden, _ = model.decode(torch.unsqueeze(decoded_output[step], dim=0), dec_hidden,
                                                    encoder_output, encoder_mask)

                if step == 0:
                    logits = F.log_softmax(logits, dim=-1).cpu()
                    probs, preds = torch.topk(logits, k, dim=-1)
                    preds = preds.view(1, -1)

                    decoded_output = torch.repeat_interleave(decoded_output, k, dim=1)
#                     dec_hidden = list(dec_hidden)
#                     dec_hidden[0] = torch.repeat_interleave(dec_hidden[0], k, dim=1)
                    dec_hidden = torch.repeat_interleave(dec_hidden, k, dim=1)
#                     dec_hidden = tuple(dec_hidden)
                    encoder_output = torch.repeat_interleave(encoder_output, k, dim=1)
                    encoder_mask = torch.repeat_interleave(encoder_mask, k, dim=1)


                    decoded_output[step+1] = preds.long()
                    eos_tokens = preds == eos_id
                    decoded_probs = torch.zeros_like(decoded_output).float()
                    decoded_probs[step+1] = probs.view(1, -1)
                    beam_probs = torch.sum(decoded_probs, dim=0, keepdims=True)

                else: 
                    if torch.sum(eos_tokens) == eos_tokens.shape[1]: 
                        break

                    new_tensor = -1e9*torch.ones(len(enc_for_s2s)+1).float().cuda()
                    new_tensor[pad_id] = 1e9
#                     import ipdb
#                     ipdb.set_trace()
#                     if torch.sum(eos_tokens) >0:
                    logits[eos_tokens] = new_tensor
                    logits = F.log_softmax(logits, dim=-1).cpu()
                    beam_probs = logits.permute(2, 0, 1) + beam_probs
                    beam_probs = beam_probs.permute(1, 2, 0)  # back to 1, 320, 8000

                    beam_probs = beam_probs.view(1, beam_probs.shape[1]//k, -1)
                    beam_probs, top_beams = torch.topk(beam_probs, k, dim=-1)
                    beam_probs = beam_probs.view(1, -1)

                    tokens = (top_beams%(len(enc_for_s2s)+1))
                    tokens = tokens.view(1, -1)

                    try: 
                        beams = (top_beams//(len(enc_for_s2s)+1))
                        beams = beams.permute(2, 0, 1)
                        beams = torch.squeeze(beams, dim=1)
                        beams += torch.arange(beams.shape[-1])*k # get them back into the beamshape.
                        beams = beams.permute(1, 0).view(-1)
                    except Exception: 
                        print(beams.shape, top_beams.shape)

                    decoded_output = torch.index_select(decoded_output, 1, beams)
                    decoded_output[step+1] = tokens
#                     dec_hidden = list(dec_hidden)
                    dec_hidden = torch.index_select(dec_hidden, 1, beams.cuda())
#                     dec_hidden[1] = torch.index_select(dec_hidden[1], 1, beams.cuda())
                    encoder_output = torch.index_select(encoder_output, 1, beams.cuda())
                    encoder_mask = torch.index_select(encoder_mask, 1, beams.cuda())
                    eos_tokens = torch.index_select(eos_tokens, 1, beams)
                    eos_tokens = torch.logical_or(eos_tokens, tokens == eos_id)

            # Implementation tip: once an EOS token has been generated, force the output
            # for that example to be padding tokens in all subsequent time steps by
            # adding a large positive number like 1e9 to the appropriate logits.
    
            beam_probs = beam_probs.view(1, -1, k)
            which_beam = torch.argsort(beam_probs, dim=-1, descending=True)
            decoded_output = decoded_output.view(decoded_output.shape[0], -1, k)
    #         print(decoded_output.shape, which_beam.shape)
            for kk, (order, decoded_sents) in enumerate(zip(torch.squeeze(which_beam, dim=0), decoded_output.permute(1, 0, 2))): 
                cand_sentences = []
                for i in order:
                    decoded_sent = decoded_sents[:, i]
                    string_form = []
                    for d in decoded_sent[1:]:
                        d = d.item()
                        string_form.append(d)
                        if d == eos_id: 
                            break
                    cand_sentences.append([rev_enc[i] for i in string_form])
                sentences.append(cand_sentences)
                gts.append(gt_text[inds[kk]])
                gtp.append(gt_ph[inds[kk]])
        return sentences, gts, gtp
    # YOUR CODE HERE

def convert_phones(phone_list, rev_dict):
    if phone_list.startswith('_'):
        phone_list = phone_list[1:]
    if phone_list.startswith('|'):
        phone_list = phone_list[1:]
    if phone_list.startswith('_'):
        phone_list = phone_list[1:]
        
    results = ['']
    for w in (phone_list.split('|')):
#         print('w', w)
        w = w.replace('_', ' ').strip() + ' |'
        if w == ' |': 
            continue
        
        cand_words = rev_dict.get(w, '<unk>')
        nr = []
        for r in results: 
            for c in cand_words: 
                nr.append((r + ' ' + c).strip())
        results = nr
    return results

def calc_beam_per_s2s(model, test_loader, gt_text, labels, enc_for_s2s, beamwidth=30):

    sents, gs, ps = predict_beam_s2s(model, test_loader, k=beamwidth, gt_text=gt_text, gt_ph =labels['ph_label'].values, enc_for_s2s=enc_for_s2s)

    net_per = []
    for s, p in zip(sents, ps):
        per = edit_distance(s[0][:-1], p)/len(p)
        net_per.append(per)

    return np.mean(net_per)