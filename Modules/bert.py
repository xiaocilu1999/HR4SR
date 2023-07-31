import math

import torch
from torch import nn
import torch.nn.functional as F

from Modules.utils import PositionedFeedForward, SublayerConnection


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(query, key, value, mask=None, dropout=None):
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(query.size(-1))
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        p_attention = F.softmax(scores, -1)
        if dropout is not None:
            p_attention = dropout(p_attention)
        return torch.matmul(p_attention, value), p_attention


class MultiHeadedAttention(nn.Module):

    def __init__(self, head, model_dimension, dropout):
        super().__init__()
        assert model_dimension % head == 0
        self.d_k = model_dimension // head
        self.head = head
        self.linear_layers = nn.ModuleList([nn.Linear(model_dimension, model_dimension) for _ in range(3)])
        self.output_linear = nn.Linear(model_dimension, model_dimension)
        self.attention = Attention()
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)
        query, key, value = [linear(x).view(batch_size, -1, self.head, self.d_k).transpose(1, 2) for linear, x in
                             zip(self.linear_layers, (query, key, value))]
        x, attention = self.attention(query, key, value, mask, self.dropout)
        x = x.transpose(1, 2).contiguous().view(batch_size, -1, self.head * self.d_k)
        return self.output_linear(x)


class TransformerBlock(nn.Module):
    def __init__(self, args):
        super().__init__()
        hidden_size = args.bert_hidden_size
        attention_heads = args.bert_num_heads
        feed_forward_hidden_size = args.bert_hidden_size * 4
        dropout = args.bert_dropout

        self.attention = MultiHeadedAttention(attention_heads, hidden_size, dropout)
        self.feed_forward = PositionedFeedForward(hidden_size, feed_forward_hidden_size, dropout)
        self.input_sublayer = SublayerConnection(hidden_size, dropout)
        self.output_sublayer = SublayerConnection(hidden_size, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.input_sublayer(x, lambda _x: self.attention.forward(_x, _x, _x, mask))
        x = self.output_sublayer(x, self.feed_forward)
        return self.dropout(x)


class BERT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.bert_num_blocks = args.bert_num_blocks
        self.transformer_blocks = nn.ModuleList([TransformerBlock(args) for _ in range(self.bert_num_blocks)])

    def encoder(self, x, attention_mask=None):
        for transformer in self.transformer_blocks:
            x = transformer(x, attention_mask)
        return x
