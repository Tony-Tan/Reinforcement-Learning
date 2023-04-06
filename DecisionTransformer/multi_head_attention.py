import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, model_dims, queries_keys_dims, value_dims, masked=False):
        super(MultiHeadAttention, self).__init__()
        self.model_dims = model_dims
        self.masked = masked
        self.queries_keys_dims = queries_keys_dims
        self.value_dims = value_dims
        self.query_projections = nn.Linear(self.heads_num * self.queries_keys_dims, self.heads_num * self.model_dims,
                                           bias=False)
        self.key_projections = nn.Linear(self.heads_num * self.queries_keys_dims, self.heads_num * self.model_dims,
                                         bias=False)
        self.value_projections = nn.Linear(self.heads_num * self.value_dims, self.heads_num * self.model_dims,
                                           bias=False)
        self.multi_head_output = nn.Linear(self.heads_num * self.value_dims, self.model_dims, bias=False)


class Transformer(torch.nn.Module):
    def __init__(self, heads_num, layers_num):
        super(Transformer, self).__init__()
        self.heads_num = heads_num
        self.layers_num = layers_num

    def forward(self, values, keys, queries):
        pass
