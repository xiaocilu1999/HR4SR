import math

import torch
from torch import nn

from Utils import fix_random_seed_as
from Modules.bert import BERT
from Modules.embedding import ItemEmbedding, PositionalEmbedding, TimestepEmbedding


class DiffusionBERT(nn.Module):
    def __init__(self, args, item_count):
        super().__init__()

        fix_random_seed_as(0)

        self.device = args.device
        self.sequence_max_length = args.sequence_max_length
        self.bert_hidden_size = args.bert_hidden_size
        self.max_step = args.max_step
        self.dataset_name = args.dataset_name
        self.bert_dropout = args.bert_dropout
        self.increment = args.increment
        self.item_embedding = ItemEmbedding(item_count + 2, self.bert_hidden_size)
        self.position_embedding = PositionalEmbedding(self.sequence_max_length, self.bert_hidden_size)
        self.time_embedding = TimestepEmbedding(self.max_step, self.bert_hidden_size)
        self.bert_1 = BERT(args)
        self.bert_2 = BERT(args)

        self.dropout = nn.Dropout(self.bert_dropout)
        self.layernorm = nn.LayerNorm([self.sequence_max_length, self.bert_hidden_size])
        self.out = nn.Linear(self.bert_hidden_size, item_count + 1)

    def forward(self, sequence):
        item_embedding = self.item_embedding(sequence)
        position_embedding = self.position_embedding(sequence)
        attention_mask = (sequence > 0).unsqueeze(1).repeat(1, sequence.size(1), 1).unsqueeze(1)
        diffusion_steps = torch.randint(0, self.max_step, size=(sequence.shape[0],)).to(self.device)
        time_embedding = self.time_embedding(diffusion_steps)

        feature_input = self.dropout(self.layernorm(item_embedding + position_embedding))
        feature_output = self.bert_1.encoder(feature_input, attention_mask)

        with torch.no_grad():
            noise = torch.randn_like(item_embedding)
            alpha = 1 - torch.sqrt((diffusion_steps + 1) / self.max_step).view(-1, 1, 1)
            noisy_item = torch.sqrt(alpha) * feature_output + torch.sqrt(1 - alpha) * noise

        denoise_input = self.dropout(self.layernorm(noisy_item + time_embedding))
        denoise_output = self.bert_2.encoder(denoise_input, attention_mask)

        denoise_output = self.out(denoise_output)
        feature_output = self.out(feature_output)

        return denoise_output, feature_output

    def sampler(self, sequence, noisy_item=None):
        denoise_output = None
        item_embedding = self.item_embedding(sequence)
        position_embedding = self.position_embedding(sequence)
        attention_mask = (sequence > 0).unsqueeze(1).repeat(1, sequence.size(1), 1).unsqueeze(1)
        shape = [sequence.shape[0], self.sequence_max_length, self.bert_hidden_size]
        if noisy_item is None:
            noisy_item = torch.normal(0, 1, shape).to(self.device)
            save_noisy = noisy_item
        else:
            save_noisy = noisy_item

        feature_input = self.dropout(self.layernorm(item_embedding + position_embedding))
        feature_output = self.bert_1.encoder(feature_input, attention_mask)

        for t in range(self.max_step - 1, 0, -self.increment):
            diffusion_steps = torch.ones(size=(sequence.shape[0],), device=self.device).long() * t
            time_embedding = self.time_embedding(diffusion_steps)
            denoise_input = self.dropout(self.layernorm(noisy_item + time_embedding))
            denoise_output = self.bert_2.encoder(denoise_input, attention_mask)
            # DDIM
            if t + 1 - self.increment >= 0:
                alpha_tk = 1 - math.sqrt((t + 1 - self.increment) / self.max_step)  # +1e-5
                alpha_t = 1 - math.sqrt((t + 1) / self.max_step) + 1e-5
                noise = (noisy_item - math.sqrt(alpha_t) * denoise_output) / math.sqrt(1 - alpha_t)
                noisy_item = math.sqrt(alpha_tk) * (noisy_item / math.sqrt(alpha_t) + (
                            math.sqrt((1 - alpha_tk) / alpha_tk) - math.sqrt((1 - alpha_t) / alpha_t)) * noise)

        feature_output = self.out(feature_output)
        denoise_output = self.out(denoise_output)

        return denoise_output, feature_output, save_noisy
