import torch
import torchaudio
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor):
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        indices = torch.unique_consecutive(indices, dim=-1)
        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [num_seq,]
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        joined = [self.labels[i] for i in indices]
        return joined
    
    
def batch_wer_cer(emission, gts, gt_phones, greedy, beam_search_decoder):
    batch_hypotheses = beam_search_decoder(emission.cpu())
    import ipdb 
    ipdb.set_trace()
    transcripts = [" ".join(hypo[0].words) for hypo in batch_hypotheses]
    tokens = [hypo[0].tokens for hypo in batch_hypotheses]
    net_wer = 0
    net_cer = 0
    net_per = 0
#     import ipdb 
#     ipdb.set_trace()
    for beam_search_transcript, gt, gt_phonemes, trans_phones in zip(transcripts, gts, gt_phones, tokens):
 
        trans_phones = [greedy.labels[i] for i in trans_phones]
        net_per += torchaudio.functional.edit_distance(gt_phonemes[1:], trans_phones[1:]) / (len(gt_phonemes)-1)
        net_cer += torchaudio.functional.edit_distance(gt, beam_search_transcript)/len(gt)
        net_wer += torchaudio.functional.edit_distance(gt.split(' '), beam_search_transcript.split(' '))/len(gt.split(' ')) 
        
    return net_per, net_cer, net_wer, transcripts
    
    
def greedy_beam_wer_cer(emission, gt, gt_phonemes, greedy, beam_search_decoder, print_hypo=False): 
    """
    Does beam/greedy search and tells us how we're doing. 
    
    Inputs: 
        emission - the probabilities of each phoneme at each timestep
        gt - the text of the ground truth sentence
        gt_phonemes, the list of the ground truth phoneme sequence
        beam_search_decoder, at torchaudio.models.decoder ctc_decoder
        
        
    Outputs: 
        Greedy phone error rate
        The ground truth text
        the beam search text
        beam search char error rate
        beam search wer
        beam search phone error rate
    """
#     import ipdb 
#     ipdb.set_trace()
    greedy_result = greedy(emission)
    greedy_per = torchaudio.functional.edit_distance(gt_phonemes[1:], greedy_result[1:]) / (len(gt_phonemes)-1)
    beam_search_result = beam_search_decoder(emission.cpu().unsqueeze(0))
    if len(beam_search_result) > 0: 
        
        if print_hypo:
            print('gt:', gt)
            print('top 50')
            print(len(beam_search_result[0]))
            for b in beam_search_result[0][:50]:
                print(b.words)
            print('---')
        try:
            beam_search_transcript = " ".join(beam_search_result[0][0].words).strip()
        except Exception: 
            beam_search_transcript = "nobeam"
    else: 
        beam_search_transcript = 'nobeams'
    try:
        beam_search_phones = [greedy.labels[i] for i in beam_search_result[0][0].tokens]
    except Exception: 
        beam_search_phones = 'AH'
        
    # ignore silence token at the start
    beam_search_per = torchaudio.functional.edit_distance(gt_phonemes[1:], beam_search_phones[1:]) / (len(gt_phonemes) -1)
    beam_cer = torchaudio.functional.edit_distance(gt, beam_search_transcript)/len(gt)
    beam_wer = torchaudio.functional.edit_distance(gt.split(' '), beam_search_transcript.split(' '))/len(gt.split(' ')) 
    return greedy_per, gt, beam_search_transcript, beam_cer, beam_wer, beam_search_per