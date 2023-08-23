import torch
import torch.nn as nn


class CnnRnnClassifier(torch.nn.Module):
    """
    The CNN RNN classifier that I used for the bravo1 decoding, in pytorch.
    """

    def __init__(self, rnn_dim, KS, num_layers, dropout, n_classes, bidirectional, in_channels=128, keeptime=False,
                 token_input=None):
        super().__init__()

        self.preprocessing_conv = nn.Conv1d(in_channels=in_channels,
                                            out_channels=rnn_dim,
                                            kernel_size=KS,
                                            stride=KS)

        self.BiGRU = nn.GRU(input_size=rnn_dim, hidden_size=rnn_dim,
                            num_layers=num_layers,
                            bidirectional=bidirectional,
                            dropout=dropout)
        self.num_layers = num_layers
        self.rnn_dim = rnn_dim
        self.keeptime = keeptime

        self.dropout = nn.Dropout(dropout)
        if bidirectional:
            mult = 2
        else:
            mult = 1
        self.mult = mult
        if keeptime:
            self.postprocessing_conv = nn.ConvTranspose1d(in_channels=rnn_dim * mult,
                                                          out_channels=rnn_dim * mult,
                                                          kernel_size=KS,
                                                          stride=KS)
        self.dense = nn.Linear(rnn_dim * mult, n_classes)

    def forward(self, x):
        # x comes in bs, t, c
        x = x.contiguous().permute(0, 2, 1)
        # now bs, c, t
        x = self.preprocessing_conv(x)
        #         x = F.relu(x)
        x = self.dropout(x)
        x = x.contiguous().permute(2, 0, 1)
        # now t, bs, c
        output, x = self.BiGRU(x)  # output: t,bs,d*c
        if not self.keeptime:
            x = x.contiguous().view(self.num_layers, self.mult, -1, self.rnn_dim)
            # (2, bs, rnn_dim)
            x = x[-1]  # Only care about the output at the final layer.
            # (2, bs, rnn_dim)
            x = x.contiguous().permute(1, 0, 2)
            x = x.contiguous().view(x.shape[0], -1)
        else:
            x = output.contiguous().permute(1, 2, 0)  # bs,d*c,t
            x = self.postprocessing_conv(x)
            x = x.permute(0, 2, 1)  # bs,t,d*c

        x = self.dropout(x)
        out = self.dense(x)
        return out
