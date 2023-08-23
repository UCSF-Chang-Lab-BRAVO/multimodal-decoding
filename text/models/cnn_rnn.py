import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from speechbrain.nnet.RNN import LiGRU_Layer


class FlexMLP(torch.nn.Module): 
    """
    Pytorch model that has an rnn that outputs a prediction at every timepoint
    
    Uses a CNN/RNN architecture and pytorch's pack padded sequence and pad packed sequence functions. 

    
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=506):
        super().__init__()
        print('2 input convs')
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.preprocessing_conv2 = nn.Conv1d(
            in_channels=rnn_dim, 
            out_channels=rnn_dim, 
            kernel_size=KS, 
            stride=1, 
            padding='same'
        )
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        self.relu = nn.LeakyReLU()
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//self.ks
        # Bs, C, T for conv
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(self.relu(x))
        x = self.preprocessing_conv2(x)
        
        
        # reshape for RNN. 
        x = x.contiguous().permute(2, 0, 1)
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, hiddens = self.BiGRU(packed)
        unpacked_emissions, lens_unpacked = pad_packed_sequence(emissions)
        unpacked_outputs = self.dense(unpacked_emissions)
        return unpacked_outputs, lens_unpacked
    
class FlexibleCnnRnnClassifier(torch.nn.Module): 
    """
    Pytorch model that has an rnn that outputs a prediction at every timepoint
    
    Uses a CNN/RNN architecture and pytorch's pack padded sequence and pad packed sequence functions. 

    
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=506):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//self.ks
        # Bs, C, T for conv
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        
        # reshape for RNN.  T, B, C
        x = x.contiguous().permute(2, 0, 1)
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, hiddens = self.BiGRU(packed)
        unpacked_emissions, lens_unpacked = pad_packed_sequence(emissions)
        unpacked_outputs = self.dense(unpacked_emissions)
        return unpacked_outputs, lens_unpacked
    

class FlexibleLiGRUClassifier(torch.nn.Module): 
    """
    Pytorch model that has an rnn that outputs a prediction at every timepoint
    
    Uses a CNN/RNN architecture and pytorch's pack padded sequence and pad packed sequence functions. 

    
    """
    def __init__(self, rnn_dim, KS, num_layers, batch_size, dropout, n_targ, bidirectional, in_channels=506):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = LiGRU_Layer(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers, 
                                 batch_size=batch_size,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//self.ks
        # Bs, C, T for conv
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        
        # reshape for RNN. 
        x = x.contiguous().permute(2, 0, 1)
#         packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        unpacked_emissions = self.BiGRU(x)
        lens_unpacked = lens.int().cpu()
        unpacked_outputs = self.dense(unpacked_emissions)
        return unpacked_outputs, lens_unpacked
    
    
class FlexLSTM(torch.nn.Module): 
    """
    Pytorch model that has an rnn that outputs a prediction at every timepoint
    
    Uses a CNN/RNN architecture and pytorch's pack padded sequence and pad packed sequence functions. 

    
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, in_channels=506):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=KS)
        self.BiGRU = nn.LSTM(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense = nn.Linear(rnn_dim*mult, n_targ)
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//self.ks
        # Bs, C, T for conv
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        
        # reshape for RNN. 
        x = x.contiguous().permute(2, 0, 1)
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, hiddens = self.BiGRU(packed)
        unpacked_emissions, lens_unpacked = pad_packed_sequence(emissions)
        unpacked_outputs = self.dense(unpacked_emissions)
        return unpacked_outputs, lens_unpacked
    
    
class CRDNN(torch.nn.Module): 
    """
    Pytorch model that has an rnn that outputs a prediction at every timepoint
    
    Uses a CNN/RNN architecture and pytorch's pack padded sequence and pad packed sequence functions. 
    Uses a CRDNN

    
    """
    def __init__(self, rnn_dim, KS, num_layers, dropout, n_targ, bidirectional, stride1 = 2, 
                 stride2= 1, in_channels=506, activation=nn.LeakyReLU(), KS2 = None):
        super().__init__()
        
        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                           out_channels=rnn_dim,
                                           kernel_size=KS,
                                           stride=stride1)
        
        if KS2 is None: 
            KS2 = KS
        self.conv2 = nn.Conv1d(in_channels=rnn_dim, out_channels=rnn_dim, kernel_size=KS2,
                              stride=stride2)
        
        self.act = activation
        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim, 
                           num_layers =num_layers,
                            bidirectional=bidirectional, 
                            dropout=dropout)
        
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.ks = KS
        
        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else: 
            mult = 1
        self.mult = mult
        self.dense1 = nn.Linear(rnn_dim*mult, rnn_dim)
#         self.dense2 = nn.Linear(rnn_dim, rnn_dim)
        self.dense2 = nn.Linear(rnn_dim, n_targ)
        
    def forward(self, x, lens): 
        # x comes in bs, t, c
        lens = lens//(self.stride1 * self.stride2)
        # Bs, C, T for conv
        x = x.contiguous().permute(0, 2, 1)
        x = self.preprocessing_conv(x)
        x = self.dropout(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.dropout(x)
        x = self.act(x)
        
        # reshape for RNN. 
        x = x.contiguous().permute(2, 0, 1)
        packed = pack_padded_sequence(x, lens.int().cpu(), enforce_sorted=False)
        emissions, hiddens = self.BiGRU(packed)
        unpacked_emissions, lens_unpacked = pad_packed_sequence(emissions)
        unpacked_outputs_ = self.dropout(self.act(self.dense1(unpacked_emissions)))
        unpacked_outputs = self.dense2(unpacked_outputs_)
        return unpacked_outputs, lens_unpacked