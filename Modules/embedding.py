from torch import nn


class ItemEmbedding(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super().__init__()
        self.ie = nn.Embedding(vocab_size, embed_size, 0)
        self.weight = self.ie.weight

    def forward(self, x):
        return self.ie(x)


class PositionalEmbedding(nn.Module):
    def __init__(self, max_len, model_dimension):
        super().__init__()
        self.pe = nn.Embedding(max_len, model_dimension)

    def forward(self, x):
        batch_size = x.size(0)
        return self.pe.weight.unsqueeze(0).repeat(batch_size, 1, 1)


class TimestepEmbedding(nn.Module):
    def __init__(self, steps, embed_size):
        super().__init__()
        self.te = nn.Embedding(steps, embed_size)
        nn.init.constant_(self.te.weight, 0)

    def forward(self, x):
        return self.te(x).unsqueeze(1)
