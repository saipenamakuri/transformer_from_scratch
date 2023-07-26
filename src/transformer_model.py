import torch
from torch import nn
import math
import torch


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

    def attention(self, Q, K, V, mask):
        scale = math.sqrt(self.d_k)

        # (B, num_heads, S or T, d_k), (B, num_heads, S or T, d_k) -> (B, num_heads, S or T, S or T)
        scores = torch.matmul(Q, K.transpose(-2, -1))

        # Mask the scores given a Boolean Mask
        # Dim may vary based on use case
        # Causal Masking in Self Attention in Decoder: (B, 1, T, T)
        # Masking Pad Tokens: (B, 1, 1, S)
        # Gets broadcasted to shape of attention weights and masks where mask = 1
        scores = scores.masked_fill(mask, float("-inf"))

        # Softmax along last dimension. Intuitively every element from the Target sequence
        # gets a distribution over the Source sequence.
        attention_weights = nn.functional.softmax(scores / scale, dim=-1)

        # (B, num_heads, T, S), (B, num_heads, S, d_v) -> (B, num_heads, T, d_v)
        attention = torch.matmul(attention_weights, V)

        return attention, attention_weights

    def forward(self, Q, K, V, mask):
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
        attention_out, attention_weights = self.attention(Q_proj, K_proj, V_proj, mask)

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
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)

    def forward(self, inp, mask):
        # (B, T, d_model) -> (B, T, d_model)
        inp2 = self.layer_norm1(self.mha(Q=inp, K=inp, V=inp, mask=mask) + inp)
        out = self.layer_norm2(self.ff(inp2) + inp2)

        return out


class Encoder:
    def __init__(self, d_model, d_ff, num_heads, num_encoders):
        self.encoder_list = [
            EncoderLayer(d_model, d_ff, num_heads) for _ in range(num_encoders)
        ]

    def forward(self, inp, mask):
        res = inp
        for encoder in self.encoder_list:
            res = encoder(res, mask)
        return res


class DecoderLayer:
    def __init__(self, d_model, d_ff, num_heads):
        self.masked_mha = MultiHeadAttention(d_model, num_heads)
        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ff = PositionWiseFeedForwardNet(d_ff, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, inp, enc_out, target_mask, source_mask):

        inp2 = self.norm1(self.masked_mha(Q=inp, K=inp, V=inp, mask=target_mask) + inp)
        inp3 = self.norm2(
            self.mha(Q=inp2, K=enc_out, V=enc_out, mask=source_mask) + inp2
        )
        out = self.norm3(self.ff(inp3) + inp3)

        return out


class Decoder:
    def __init__(self, d_model, d_ff, num_heads, num_decoders):
        self.decoder_list = [
            DecoderLayer(d_model, d_ff, num_heads) for _ in range(num_decoders)
        ]

    def forward(self, inp, enc_out, source_mask, target_mask):

        res = inp
        for dec in self.decoder_list:
            res = dec(res, enc_out, target_mask, source_mask)

        return res


class Transformer:
    def __init__(
        self, d_model, d_ff, num_heads, num_encoders, num_decoders, num_encoder_tokens, num_decoder_tokens
    ):
        self.decoder = Decoder(d_model, d_ff, num_heads, num_decoders)
        self.encoder = Encoder(d_model, d_ff, num_heads, num_encoders)
        self.position_encoder = PositionalEncoding(d_model=d_model)
        self.encoder_embedding = nn.Embedding(num_encoder_tokens, d_model)
        self.decoder_embedding = nn.Embedding(num_decoder_tokens, d_model)
        self.linear = nn.Linear(d_model, num_decoder_tokens)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, inp, out, source_mask, target_mask):
        # inp will be a Tensor of size (B, T), each element is a token index

        inp_embedding = self.embedding(inp)
        out_embedding = self.embedding(out)

        inp_embedding = self.position_encoder(inp_embedding)
        out_embedding = self.position_encoder(out_embedding)

        enc_out = self.encoder(inp_embedding, source_mask)
        dec_out = self.decoder(out_embedding, enc_out, source_mask, target_mask)

        # (B, T, d_model) -> (B, T, num_tokens)
        out = self.linear(dec_out)
        out = self.softmax(out)

        return out
