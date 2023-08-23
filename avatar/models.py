import torch
import torch.nn as nn
import torch.nn.functional as F
    
class MLP(nn.Module):
    def __init__(self, n_hidden, input_shape,  n_classes, dropout, num_layers=2):
        super().__init__()
        print('is', input_shape)
        self.linear1 = nn.Linear(input_shape, n_hidden)
        self.act = nn.ReLU()
        if num_layers > 2: 
            self.extras = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(num_layers-2)])
        self.num_layers = num_layers
        self.linear3 = nn.Linear(n_hidden, n_classes)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(self.act(self.linear1(x)))
        if self.num_layers > 2: 
            for lay in self.extras:
                x = self.dropout(lay(x))
        x = self.linear3(x)
        return x
