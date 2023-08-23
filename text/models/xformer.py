import math
import torch
import torch.nn as nn
import torch.nn.functional as F
class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 1000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class SpeechXFormer(nn.Module):
    def __init__(self, in_channels=506, width=384, heads=4,  layers=4, conv_layers=2, out_chans=41, dropout=0.1, 
                ks=3):
        super().__init__()
        self.conv1 = torch.nn.Conv1d(in_channels, width, kernel_size=ks, stride=ks)
#         self.conv2 = torch.nn.Conv1d(width, width, kernel_size=3, stride=2)
#         self.act = torch.nn.GELU()
        encoder_layers = nn.TransformerEncoderLayer(d_model=width, nhead=heads, dropout=dropout, batch_first=True)
        self.pos_encoder = PositionalEncoding(d_model=width, max_len=1800)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, layers)
        self.lin_out = nn.Linear(width, out_chans)
        self.ks =ks
        
    def forward(self, x, lens):
#         print(x.shape)
        lens = lens//self.ks
        x = x.contiguous().permute(0, 2, 1) # B, L, C
        x = self.conv1(x)
#         print(x.shape)
        # B< C< L 
        x = x.permute(2, 0, 1)
        x = self.transformer_encoder(x) # B C, L
        x = self.lin_out(x) # LBC baby. 
#         print(x.shape)
        return x, lens
        
        