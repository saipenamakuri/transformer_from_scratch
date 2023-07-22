import torch
from torch import nn
import math
import torch
import numpy as np

torch.nn.Transformer


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()

        assert d_model % num_heads == 0, "Num heads must cleanly divide d_model!"
        self.d_k = self.d_v = int(d_model / num_heads)
        self.num_heads = num_heads

        # d_model -> d_model because we stack weights of all heads into one weight.
        # Each head does d_model -> d_k
        self.Q_proj_net = nn.Linear(d_model, d_model)
        self.K_proj_net = nn.Linear(d_model, d_model)
        self.V_proj_net = nn.Linear(d_model, d_model)
        self.out_proj_net = nn.Linear(d_model, d_model)

    def attention(self, Q_proj, K_proj, V_proj):
        scale = math.sqrt(self.d_k)

        # (B, num_heads, T, d_k), (B, num_heads, S, d_k) -> (B, num_heads, T, S)
        attention_weights = torch.matmul(Q_proj, K_proj.transpose(-2, -1))

        # Mask out target (Q_proj) and Source (K_proj)

        # Softmax along last dimension. Intuitively every element from the Target sequence
        # gets a distribution over the Source sequence.
        attention_weights = nn.functional.softmax(attention_weights / scale, dim=-1)

        # (B, num_heads, T, S), (B, num_heads, S, d_v) -> (B, num_heads, T, d_v)
        attention = torch.matmul(attention_weights, V_proj)

        return attention, attention_weights

    def forward(self, Q, K, V, source_mask, target_mask):
        B = Q.shape[0]
        T = Q.shape[1]
        S = K.shape[1]

        # (B, T, d_model) -> (B, T, d_model) -> (B, T, num_heads, d_k) -> (B, num_heads, T, d_k)
        Q_proj = self.Q_proj_net(Q).view(B, T, self.num_heads, self.d_k).transpose(1, 2)

        # (B, S, d_model) -> (B, S, d_model) -> (B, S, num_heads, d_k) -> (B, num_heads, S, d_k)
        K_proj = self.K_proj_net(K).view(B, S, self.num_heads, self.d_k).transpose(1, 2)

        # (B, S, d_model) -> (B, S, d_model) -> (B, S, num_heads, d_v) -> (B, num_heads, S, d_v)
        V_proj = self.V_proj_net(V).view(B, S, self.num_heads, self.d_v).transpose(1, 2)

        # Get the output of attention along with attention weights
        attention_out, attention_weights = self.attention(Q_proj, K_proj, V_proj)

        # (B, num_heads, T, d_v) -> (B, T, num_heads * d_v = d_model)
        attention_out = attention_out.reshape(B, -1, self.num_heads * self.d_v)

        # (B, T, d_model) -> (B, T, d_model)
        out = self.out_proj_net(attention_out)

        return out, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, max_length=5000, d_model=512):
        mat = torch.zeros(max_length, d_model)
        values = torch.arange(max_length).unsqueeze(1) * torch.pow(
            10000, -torch.arange(0, d_model, 2) / d_model
        )

        # Even Dimensions use sine
        mat[:, 0::2] = torch.sin(values)

        # Odd Dimensions use cosine
        mat[:, 1::2] = torch.cos(values)

        self.mat = mat

    def forward(self, inp):
        B, T, d_model = inp.shape

        # (T or S, d_model)
        pos_encoding = self.mat[:T]

        assert pos_encoding.shape[1] == d_model, f"{pos_encoding.shape[1]} != {d_model}"

        # Add (T or S, d_model) + (B, T or S, d_model) -> (B, T or S, d_model)
        out = pos_encoding + inp

        return out


class PositionWiseFeedForwardNet:
    def __init__(self, d_ff, d_model):
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.relu = nn.Relu()

    def forward(self, x):
        # (B, T, d_model) -> (B, T, d_ff) -> (B, T, d_model)

        x_proj = self.linear1(x)
        x_proj = self.relu(x_proj)
        out = self.linear2(x_proj)

        return out


class EncoderLayer:
    def __init__(self, d_model, d_ff, num_heads):
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForwardNet(d_ff, d_model, num_heads)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, inp):
        # (B, T, d_model) -> (B, T, d_model)
        inp2 = self.layer_norm(self.mha(Q=inp, K=inp, V=inp) + inp)
        out = self.layer_norm(self.ff(inp2) + inp2)

        return out


class Encoder:
    def __init__(self, d_model, d_ff, num_heads, num_encoders):
        self.encoder_list = [
            EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_encoders)
        ]

    def forward(self, inp):
        res = inp
        for encoder in self.encoder_list:
            res = encoder(res)
        return res