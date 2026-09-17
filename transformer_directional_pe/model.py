import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .positional_encoding import DirectionalRandomWalkPE


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention implemented from scaled dot-product attention."""

    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.output = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, return_attention=False):
        batch_size, sequence_length, _ = x.shape
        query, key, value = self.qkv(x).chunk(3, dim=-1)

        query = query.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.num_heads, self.head_dim).transpose(1, 2)

        attention_logits = query @ key.transpose(-2, -1)
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention = torch.softmax(attention_logits, dim=-1)
        attention = self.dropout(attention)

        context = attention @ value
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, self.d_model)
        output = self.output(context)

        if return_attention:
            return output, attention
        return output


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, feedforward_dim, dropout=0.0):
        super().__init__()
        self.attention_norm = nn.LayerNorm(d_model)
        self.feedforward_norm = nn.LayerNorm(d_model)
        self.attention = MultiHeadSelfAttention(d_model, num_heads, dropout)
        self.feedforward = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x, return_attention=False):
        normalized = self.attention_norm(x)
        if return_attention:
            attended, attention = self.attention(normalized, return_attention=True)
        else:
            attended = self.attention(normalized)
            attention = None
        x = x + attended
        x = x + self.feedforward(self.feedforward_norm(x))
        if return_attention:
            return x, attention
        return x


class DirectionalTransformerClassifier(nn.Module):
    """Standalone graph-token classifier using directional PE and custom MHA."""

    def __init__(
        self,
        input_dim,
        num_nodes,
        d_model,
        num_heads,
        num_layers,
        num_classes,
        walk_steps=3,
        dropout=0.1,
    ):
        super().__init__()
        self.node_embedding = nn.Linear(input_dim, d_model)
        self.directional_pe = DirectionalRandomWalkPE(num_nodes, d_model, walk_steps, dropout)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, 4 * d_model, dropout)
                for _ in range(num_layers)
            ]
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, node_features, adjacency, return_attention=False):
        tokens = self.node_embedding(node_features)
        tokens = self.directional_pe(tokens, adjacency)

        attentions = []
        for layer in self.layers:
            if return_attention:
                tokens, attention = layer(tokens, return_attention=True)
                attentions.append(attention)
            else:
                tokens = layer(tokens)

        graph_representation = tokens.mean(dim=1)
        logits = self.classifier(graph_representation)
        if return_attention:
            return logits, attentions
        return logits
