import torch.nn as nn
from .gelu import GELU

class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1, activate="gelu"):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = GELU() if activate=="gelu" else nn.ReLU()

    def forward(self, x):
        return self.w_2(self.activation(self.dropout(self.w_1(x))))
