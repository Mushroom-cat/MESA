import torch.nn as nn
from .single import Attention
from torch.nn import Parameter
import torch


class LinearForMeta_scaling(nn.Module):

    def __init__(self, in_dim, out_dim, lam=0.01):
        super().__init__()

        self.base_w = Parameter(torch.zeros(in_dim, out_dim))
        self.b = Parameter(torch.zeros(out_dim))
        self.lam = lam
        nn.init.normal_(self.b, std=0.01)
        nn.init.xavier_normal_(self.base_w)

    def forward(self, x, w):

        w1 = torch.mul(w, self.base_w)
        x1 = torch.matmul(x, w1)
        x2 = x1 + self.b

        return x2

class LinearForMeta_shifting(nn.Module):

    def __init__(self, in_dim, out_dim, lam=0.01):
        super().__init__()

        self.base_w = Parameter(torch.zeros(in_dim, out_dim))
        self.b = Parameter(torch.zeros(out_dim))
        self.lam = lam
        nn.init.normal_(self.b, std=0.01)
        nn.init.xavier_normal_(self.base_w)

    def forward(self, x, w):

        w1 = self.lam * w + self.base_w
        x1 = torch.matmul(x, w1)
        x2 = x1 + self.b

        return x2

class MultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        if args.modulation_mode == 'shifting':
            self.linear_layers = nn.ModuleList([LinearForMeta_shifting(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
        if args.modulation_mode == 'scaling':
            self.linear_layers = nn.ModuleList([LinearForMeta_scaling(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
        self.output_linear = nn.Linear(d_model, d_model)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, fc_w, mask=None, output=True, stride=None, args=None, users=None):

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.linear_layers[0](query, fc_w).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)


class OriginalMultiHeadedAttention(nn.Module):
    "Take in model size and number of heads."
    def __init__(self, h, d_model, d_output, dropout=0.1, max_len=50, args=None):
        super().__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h

        self.linear_layers = nn.ModuleList([nn.Linear(d_model, d_model), nn.Linear(d_model, d_model), nn.Linear(d_model, d_model)])
        self.output_linear = nn.Linear(d_model, d_output)
        self.attention = Attention()

        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, output=True, stride=None, args=None, users=None):

        batch_size = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k

        query = self.linear_layers[0](query).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        key = self.linear_layers[1](key).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        value = self.linear_layers[2](value).view(batch_size, -1, self.h, self.d_k).transpose(1, 2)

        # 2) Apply attention on all the projected vectors in batch.
        x, attn = self.attention(query, key, value, mask=mask, dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.h * self.d_k)

        return self.output_linear(x)

