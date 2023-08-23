import torch.nn as nn
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class HybridModel(nn.Module):
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ_ctc, n_targ_s2s, bidirectional=True, in_channels=506, 
                smoothing=0.0):
        super().__init__()

        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                       out_channels=rnn_dim,
                                       kernel_size=KS,
                                       stride=KS)
        
        self.ks = KS
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.dropout = nn.Dropout(dropout)
        # Implementation tip: To create a bidirectional LSTM, you don't need to
        # create two LSTM networks. Instead use nn.LSTM(..., bidirectional=True).
        self.enc = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                               num_layers =num_layers,
                                bidirectional=bidirectional, 
                                dropout=dropout)
        self.dec = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                               num_layers =1,
                                bidirectional=False,
                                dropout=dropout)
        self.outdense = nn.Linear(rnn_dim, n_targ_s2s)
        self.ctc_dense = nn.Linear(rnn_dim*2, n_targ_ctc)
        
        # self.ctc_dense = nn.Sequential(nn.Linear(rnn_dim*2, rnn_dim), 
        #                                nn.ReLU(), 
        #                                nn.Linear(rnn_dim, n_targ_ctc))
        
        self.embedding = nn.Embedding(n_targ_s2s, rnn_dim)
        
        self.ctc_loss = nn.CTCLoss()

    def encode(self, x, lens):
        """Encode the source batch using a bidirectional LSTM encoder.

        Args:
          source: An integer tensor with shape (max_source_sequence_length,
            batch_size) containing subword indices for the source sentences.

        Returns:
          A tuple with three elements:
            encoder_output: The output of the bidirectional LSTM with shape
              (max_source_sequence_length, batch_size, 2 * hidden_size).
            encoder_mask: A boolean tensor with shape (max_source_sequence_length,
              batch_size) indicating which encoder outputs correspond to padding
              tokens. Its elements should be True at positions corresponding to
              padding tokens and False elsewhere.
            encoder_hidden: The final hidden states of the bidirectional LSTM (after
              a suitable projection) that will be used to initialize the decoder.
              This should be a pair of tensors (h_n, c_n), each with shape
              (num_layers, batch_size, hidden_size). Note that the hidden state
              returned by the LSTM cannot be used directly. Its initial dimension is
              twice the required size because it contains state from two directions.

        The first two return values are not required for the baseline model and will
        only be used later in the attention model. If desired, they can be replaced
        with None for the initial implementation.
        """
        lens = lens//self.ks
        # Bs, C, T for conv
        # print(x.shape)
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        # reshape for RNN.  T, B, C
        x = x.contiguous().permute(2, 0, 1)
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, h_n = self.enc(packed)
        unpacked_output, lens_unpacked = pad_packed_sequence(emissions)
        #         unpacked_outputs = self.dense(unpacked_emissions)

        h_n = h_n.contiguous().view(2, self.num_layers, -1, self.rnn_dim)
        h_n = h_n[0, : , :, :] + h_n[1, : , :, :]
        final_hidden = h_n
        encoder_mask = unpacked_output == 0
        
        
        # TODO: add the ctc loss here. 
        for_ctc = self.ctc_dense(unpacked_output)
        
        return unpacked_output, encoder_mask, final_hidden, for_ctc, lens
    

    def decode(self, decoder_input, initial_hidden, encoder_output, encoder_mask, device='cuda'):
        """Run the decoder LSTM starting from an initial hidden state.

        The third and fourth arguments are not used in the baseline model, but are
        included for compatibility with the attention model in the next section.

        Args:
          decoder_input: An integer tensor with shape (max_decoder_sequence_length,
            batch_size) containing the subword indices for the decoder input. During
            evaluation, where decoding proceeds one step at a time, the initial
            dimension should be 1.
          initial_hidden: A pair of tensors (h_0, c_0) representing the initial
            state of the decoder, each with shape (num_layers, batch_size,
            hidden_size).
          encoder_output: The output of the encoder with shape
            (max_source_sequence_length, batch_size, 2 * hidden_size).
          encoder_mask: The output mask from the encoder with shape
            (max_source_sequence_length, batch_size). Encoder outputs at positions
            with a True value correspond to padding tokens and should be ignored.

        Returns:
          A tuple with three elements:
            logits: A tensor with shape (max_decoder_sequence_length, batch_size,
              vocab_size) containing unnormalized scores for the next-word
              predictions at each position.
            decoder_hidden: A pair of tensors (h_n, c_n) with the same shape as
              initial_hidden representing the updated decoder state after processing
              the decoder input.
            attention_weights: This will be implemented later in the attention
              model, but in order to maintain compatible type signatures, we also
              include it here. This can be None or any other placeholder value.
        """

        # These arguments are not used in the baseline model.
        del encoder_output
        del encoder_mask
        decoder_input = decoder_input.to(device)
        decoder_emb = self.embedding(decoder_input)
        intial_hidden = torch.unsqueeze(initial_hidden[-1], dim=0)
        decoder_out, dec_hidden = self.dec(decoder_emb, initial_hidden)
        logits = self.outdense(decoder_out)
        # YOUR CODE HERE

        return logits, dec_hidden, None

    def compute_loss(self, source, lens, target, target_ctc, ctc_targ_lens, device, alpha=0.5):
        """Run the model on the source and compute the loss on the target.

        Args:
          source: An integer tensor with shape (max_source_sequence_length,
            batch_size) containing subword indices for the source sentences.
          target: An integer tensor with shape (max_target_sequence_length,
            batch_size) containing subword indices for the target sentences.

        Returns:
          A scalar float tensor representing cross-entropy loss on the current batch.
        """

        # Implementation tip: don't feed the target tensor directly to the decoder.
        # To see why, note that for a target sequence like <s> A B C </s>, you would
        # want to run the decoder on the prefix <s> A B C and have it predict the
        # suffix A B C </s>.s
        encoder_output, encoder_mask, encoder_hidden, ctc_emissions, outlens = self.encode(source, lens)
#         lengths = torch.sum(source != pad_id, axis=0)

        #### Lets do the ctc loss here. 
        ctc_emissions = F.log_softmax(ctc_emissions, dim=-1)
        ctc_loss = self.ctc_loss(ctc_emissions, target_ctc, outlens.int().cpu(), ctc_targ_lens.int().cpu())
        
        ####
    
    
    
        
        target = target.T # T, B
        decoder_input = target[:-1, :]
        decoder_targs = target[1:, :]
        logits, decoder_hidden, attn = self.decode(decoder_input, encoder_hidden, encoder_output, encoder_mask, device)
        # Logits shape will be L, BS, V, 
        logits = logits.contiguous().permute(0, 2, 1)
        loss = torch.nn.functional.cross_entropy(logits, decoder_targs.to(device), ignore_index=0)
        
        total_loss = ctc_loss * alpha + (1-alpha)*loss
        return total_loss, ctc_loss.item(), loss.item()
    
#     def return_ctc_emissions(source, lens, target, target_ctc, device):
        
        
        
    
class HybridAttn(HybridModel):
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ_ctc, n_targ_s2s, bidirectional=True, in_channels=506, smoothing=0.0):
        super().__init__(rnn_dim, KS, num_layers, dropout, n_targ_ctc, n_targ_s2s, bidirectional=True, 
                         in_channels=in_channels, smoothing=smoothing)

        self.attention_weights = nn.Linear(2*rnn_dim, rnn_dim)

    def decode(self, decoder_input, initial_hidden, encoder_output, encoder_mask, device='cuda'):
        """Run the decoder LSTM starting from an initial hidden state.

        The third and fourth arguments are not used in the baseline model, but are
        included for compatibility with the attention model in the next section.

        Args:
          decoder_input: An integer tensor with shape (max_decoder_sequence_length,
            batch_size) containing the subword indices for the decoder input. During
            evaluation, where decoding proceeds one step at a time, the initial
            dimension should be 1.
          initial_hidden: A pair of tensors (h_0, c_0) representing the initial
            state of the decoder, each with shape (num_layers, batch_size,
            hidden_size).
          encoder_output: The output of the encoder with shape
            (max_source_sequence_length, batch_size, 2 * hidden_size).
          encoder_mask: The output mask from the encoder with shape
            (max_source_sequence_length, batch_size). Encoder outputs at positions
            with a True value correspond to padding tokens and should be ignored.

        Returns:
          A tuple with three elements:
            logits: A tensor with shape (max_decoder_sequence_length, batch_size,
              vocab_size) containing unnormalized scores for the next-word
              predictions at each position.
            decoder_hidden: A pair of tensors (h_n, c_n) with the same shape as
              initial_hidden representing the updated decoder state after processing
              the decoder input.
            attention_weights: A tensor with shape (max_decoder_sequence_length,
              batch_size, max_source_sequence_length) representing the normalized
              attention weights. This should sum to 1 along the last dimension.
        """

        # Implementation tip: use a large negative number like -1e9 instead of
        # float("-inf") when masking logits to avoid numerical issues.

        # Implementation tip: the function torch.einsum may be useful here.
        # See https://rockt.github.io/2018/04/30/einsum for a tutorial.

        # YOUR CODE HERE
        decoder_input = decoder_input.to(device)
        decoder_emb = self.embedding(decoder_input)
        initial_hidden = torch.unsqueeze(initial_hidden[-1], dim=0)
        decoder_out, dec_hidden = self.dec(decoder_emb, initial_hidden)

        att = self.attention_weights(encoder_output)
        #     print(att.shape, decoder_out.shape)
        # b x n x m , b x m x p to b x n xp 
        #decoder_out = len, bs, c
        att = att.contiguous().permute(1, 2, 0) # bs, c, len_input
        decoder_o = decoder_out.contiguous().permute(1, 0, 2) #bs, len out, c
        ws = torch.bmm(decoder_o, att) # bs, len out, len in
        ws = ws.contiguous().permute(1, 0, 2) # len out, bs, len in 
        #     ws = torch.einsum('djk,kje->dje', [decoder_out, att.contiguous().permute(2,1,0)]) #contiguous().permute(2, 1, 0)])
        emask = torch.sum(encoder_mask, dim=-1)
        emask = emask ==0
        #     print(emask.shape)
        ws = ws +  (emask.T)*-1e-9
        attention_ws = F.softmax(ws, dim=-1) 

        # len out, bs, len in# att = bs, channels, len input
        attention_ws_m = attention_ws.contiguous().permute(1, 2, 0) # bs, len in, len out
        contexts = torch.bmm(att ,attention_ws_m) #bs, channels, len_out
        contexts = contexts.contiguous().permute(2, 0, 1)
        # cs = len out, bs, channels
        #     logits = self.outdense(decoder_out + contexts)
        decoder_out = decoder_out + contexts
        logits = self.outdense(decoder_out + contexts)
        
        return logits, dec_hidden, attention_ws
    
