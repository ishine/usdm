# Reference
# Modify some classes from: https://github.com/huggingface/transformers
# Modify some classes from: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS
# A function from: https://github.com/facebookresearch/xformers

import math
import torch
import torch.nn as nn
from torch import Tensor


# Reference: https://github.com/huawei-noah/Speech-Backbones/blob/main/Grad-TTS/model/diffusion.py
class SinusoidalPosEmb(torch.nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class GELUActivation(nn.Module):
    """
    Original Implementation of the GELU activation function in Google BERT repo when initially created. For
    information: OpenAI GPT's GELU is slightly different (and gives slightly different results): 0.5 * x * (1 +
    torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3)))) This is now written in C in nn.functional
    Also see the Gaussian Error Linear Units paper: https://arxiv.org/abs/1606.08415
    """

    def __init__(self, use_gelu_python: bool = False):
        super().__init__()
        if use_gelu_python:
            self.act = self._gelu_python
        else:
            self.act = nn.functional.gelu

    def _gelu_python(self, input: Tensor) -> Tensor:
        return input * 0.5 * (1.0 + torch.erf(input / math.sqrt(2.0)))

    def forward(self, input: Tensor) -> Tensor:
        return self.act(input)


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class SamePadLayer(nn.Module):
    def __init__(self, num_conv_pos_embeddings):
        super().__init__()
        self.num_pad_remove = 1 if num_conv_pos_embeddings % 2 == 0 else 0

    def forward(self, hidden_states):
        if self.num_pad_remove > 0:
            hidden_states = hidden_states[:, :, : -self.num_pad_remove]
        return hidden_states


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class PositionalConvEmbedding(nn.Module):
    def __init__(self, hidden_size, convpos_width, convpos_groups):
        super().__init__()
        self.conv = nn.Conv1d(
            hidden_size,
            hidden_size,
            kernel_size=convpos_width,
            padding=convpos_width // 2,
            groups=convpos_groups,
        )

        weight_norm = nn.utils.weight_norm
        if hasattr(nn.utils.parametrizations, "weight_norm"):
            weight_norm = nn.utils.parametrizations.weight_norm

        self.conv = weight_norm(self.conv, name="weight", dim=2)

        self.padding = SamePadLayer(convpos_width)
        self.activation = GELUActivation()

    def forward(self, hidden_states, y_mask):
        hidden_states = hidden_states.transpose(1, 2)

        hidden_states = self.conv(hidden_states)
        hidden_states = self.padding(hidden_states)
        hidden_states = self.activation(hidden_states)

        hidden_states = hidden_states.transpose(1, 2)
        return hidden_states * y_mask.unsqueeze(-1), y_mask


# Reference: https://github.com/facebookresearch/xformers/blob/main/xformers/components/attention/attention_patterns.py
def get_slopes(n: int):
    def get_slopes_power_of_2(n: int):
        start = 2 ** (-(2 ** -(math.log2(n) - 3)))
        ratio = start
        return [start * ratio ** i for i in range(n)]

    # In the paper, we only train models that have 2^a heads for some a. This function has
    # some good properties that only occur when the input is a power of 2. To maintain that even
    # when the number of heads is not a power of 2, we use this workaround.
    if math.log2(n).is_integer():
        return get_slopes_power_of_2(n)
    else:
        closest_power_of_2 = 2 ** math.floor(math.log2(n))
        return (
                get_slopes_power_of_2(closest_power_of_2)
                + get_slopes(2 * closest_power_of_2)[0::2][: n - closest_power_of_2]
        )


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class Attention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        is_decoder: bool = False,
        bias: bool = True,
        is_causal: bool = False
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        self.scaling = self.head_dim**-0.5
        self.is_decoder = is_decoder
        self.is_causal = is_causal

        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int):
        return tensor.view(bsz, seq_len, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        alibi_bias=None
    ):
        """Input shape: Batch x Time x Channel"""

        # if key_value_states are provided this layer is used as a cross-attention layer

        bsz, tgt_len, _ = hidden_states.size()

        # get query proj
        query_states = self.q_proj(hidden_states) * self.scaling

        # self_attention
        key_states = self._shape(self.k_proj(hidden_states), -1, bsz)
        value_states = self._shape(self.v_proj(hidden_states), -1, bsz)

        proj_shape = (bsz * self.num_heads, -1, self.head_dim)
        query_states = self._shape(query_states, tgt_len, bsz).view(*proj_shape)
        key_states = key_states.reshape(*proj_shape)
        value_states = value_states.reshape(*proj_shape)

        src_len = key_states.size(1)
        attn_weights = torch.bmm(query_states, key_states.transpose(1, 2))

        if attn_weights.size() != (bsz * self.num_heads, tgt_len, src_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz * self.num_heads, tgt_len, src_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, tgt_len, src_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, tgt_len, src_len)}, but is {attention_mask.size()}"
                )
            attn_weights = attn_weights.view(bsz, self.num_heads, tgt_len, src_len) + alibi_bias + attention_mask
            attn_weights = attn_weights.view(bsz * self.num_heads, tgt_len, src_len)

        attn_weights = nn.functional.softmax(attn_weights, dim=-1)
        attn_probs = nn.functional.dropout(attn_weights, p=self.dropout, training=self.training)

        attn_output = torch.bmm(attn_probs, value_states)

        if attn_output.size() != (bsz * self.num_heads, tgt_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz * self.num_heads, tgt_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.view(bsz, self.num_heads, tgt_len, self.head_dim)
        attn_output = attn_output.transpose(1, 2)

        # Use the `embed_dim` from the config (stored in the class) rather than `hidden_state` because `attn_output` can be
        # partitioned across GPUs when using tensor-parallelism.
        attn_output = attn_output.reshape(bsz, tgt_len, self.embed_dim)

        attn_output = self.out_proj(attn_output)

        return attn_output


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class FeedForward(nn.Module):
    def __init__(self, activation_dropout, hidden_size, intermediate_size, hidden_dropout):
        super().__init__()
        self.intermediate_dropout = nn.Dropout(activation_dropout)
        self.intermediate_dense = nn.Linear(hidden_size, intermediate_size)

        self.intermediate_act_fn = GELUActivation()

        self.output_dense = nn.Linear(intermediate_size, hidden_size)
        self.output_dropout = nn.Dropout(hidden_dropout)

    def forward(self, hidden_states):
        hidden_states = self.intermediate_dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        hidden_states = self.intermediate_dropout(hidden_states)

        hidden_states = self.output_dense(hidden_states)
        hidden_states = self.output_dropout(hidden_states)
        return hidden_states


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, intermediate_size, num_attention_heads, attention_dropout, activation_dropout, hidden_dropout):
        super().__init__()
        self.attention = Attention(
            embed_dim=hidden_size,
            num_heads=num_attention_heads,
            dropout=attention_dropout,
            is_decoder=False,
        )
        self.dropout = nn.Dropout(hidden_dropout)
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.feed_forward = FeedForward(activation_dropout, hidden_size, intermediate_size, hidden_dropout)
        self.final_layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)

    def forward(self, hidden_states, y_mask, attention_mask=None, alibi_bias=None):
        attn_residual = hidden_states
        hidden_states = self.attention(
            hidden_states, attention_mask=attention_mask, alibi_bias=alibi_bias
        )
        hidden_states = self.dropout(hidden_states)
        hidden_states = attn_residual + hidden_states
        hidden_states = hidden_states * y_mask.unsqueeze(-1)

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = hidden_states + self.feed_forward(hidden_states)
        hidden_states = hidden_states * y_mask.unsqueeze(-1)
        hidden_states = self.final_layer_norm(hidden_states)

        outputs = hidden_states

        return outputs * y_mask.unsqueeze(-1)


# Reference: https://github.com/huggingface/transformers/blob/main/src/transformers/models/wav2vec2/modeling_wav2vec2.py
class Transformer(nn.Module):
    def __init__(
            self,
            n_feats,
            n_tokens,
            embedding_dim,
            hidden_size,
            intermediate_size,
            num_attention_heads,
            num_hidden_layers,
            convpos_width,
            convpos_groups,
            convpos_depth,
            attention_dropout=0.0,
            activation_dropout=0.1,
            hidden_dropout=0.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.embedding_dim = embedding_dim
        self.num_heads = num_attention_heads
        self.num_hidden_layers = num_hidden_layers

        self.embed = nn.Embedding(n_tokens, embedding_dim)
        self.time_embed = SinusoidalPosEmb(hidden_size)
        self.proj_in = nn.Conv1d(2 * n_feats + embedding_dim, hidden_size, kernel_size=1)
        self.pos_conv_embeds = nn.ModuleList([PositionalConvEmbedding(hidden_size, convpos_width, convpos_groups) for _ in range(convpos_depth)])
        self.layer_norm = nn.LayerNorm(hidden_size, eps=1e-5)
        self.dropout = nn.Dropout(hidden_dropout)
        self.layers = nn.ModuleList([EncoderLayer(hidden_size, intermediate_size, num_attention_heads, attention_dropout, activation_dropout, hidden_dropout) for _ in range(num_hidden_layers)])
        self.skip_connections_layers = nn.ModuleList([nn.Linear(2 * hidden_size, hidden_size) for _ in range(num_hidden_layers // 2)])
        self.proj_out = nn.Conv1d(hidden_size, n_feats, kernel_size=1)

    def forward(
            self, x, y, cond, t, lengths
    ):
        x = self.embed(x) * math.sqrt(self.embedding_dim)
        x = torch.cat([x.transpose(1, 2).contiguous(), y, cond], dim=1)
        hidden_states = self.proj_in(x)

        y_max_length = y.shape[-1] + 1
        lengths = lengths + 1

        t = self.time_embed(t).squeeze(1).transpose(1, 2).contiguous()
        hidden_states = torch.cat([t, hidden_states], dim=-1)
        attention_mask = (torch.arange(0, y_max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        y_mask = (torch.arange(0, y_max_length, device=lengths.device).unsqueeze(0) < lengths.unsqueeze(1)).float()

        hidden_states = hidden_states.transpose(1, 2).contiguous()

        slope = torch.Tensor(get_slopes(self.num_heads)).cuda() * -1

        # Dynamically generate ALiBi bias without loops
        range_tensor = torch.arange(y_max_length, device=hidden_states.device).unsqueeze(0)
        alibi_bias = (slope.unsqueeze(-1).unsqueeze(-1) * torch.abs(range_tensor - range_tensor.T)
                      .unsqueeze(0).expand(self.num_heads, -1, -1))  # Broadcasting to create the matrix
        alibi_bias[:, :, 0] = 0  # Set bias to 0 for interactions with the first position

        # Reshape to match the attention weights
        alibi_bias = alibi_bias.unsqueeze(0).expand(y.shape[0], self.num_heads, y_max_length, y_max_length)

        if attention_mask is not None:
            # make sure padded tokens output 0
            expand_attention_mask = attention_mask.unsqueeze(-1).repeat(1, 1, hidden_states.shape[2])
            hidden_states[~ expand_attention_mask.bool()] = 0

            alibi_bias = alibi_bias * attention_mask[:, None, None, :].to(dtype=hidden_states.dtype).expand(attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1])
            # extend attention_mask
            attention_mask = 1.0 - attention_mask[:, None, None, :].to(dtype=hidden_states.dtype)
            attention_mask = attention_mask * torch.finfo(hidden_states.dtype).min
            attention_mask = attention_mask.expand(
                attention_mask.shape[0], 1, attention_mask.shape[-1], attention_mask.shape[-1]
            )

        position_residual = hidden_states
        for pos_conv_embed in self.pos_conv_embeds:
            hidden_states, y_mask = pos_conv_embed(hidden_states, y_mask)
        hidden_states = hidden_states + position_residual

        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)

        hidden_states = hidden_states * y_mask.unsqueeze(-1)
        skip_outputs = [hidden_states]

        for layer_idx, layer in enumerate(self.layers):
            if layer_idx < len(self.layers) // 2:
                hidden_states = layer(
                    hidden_states * y_mask.unsqueeze(-1), y_mask=y_mask, attention_mask=attention_mask, alibi_bias=alibi_bias
                )
                if layer_idx < len(self.layers) // 2 - 1:
                    skip_outputs.append(hidden_states)
            else:
                skip_output = skip_outputs.pop()
                hidden_states = self.skip_connections_layers[layer_idx - len(self.layers) // 2](torch.cat([hidden_states, skip_output], dim=-1))
                hidden_states = layer(
                    hidden_states * y_mask.unsqueeze(-1), y_mask=y_mask, attention_mask=attention_mask, alibi_bias=alibi_bias
                )

        assert len(skip_outputs) == 0
        hidden_states = hidden_states * y_mask.unsqueeze(-1)
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        # Combine concatenated outputs using a linear layer
        hidden_states = self.proj_out(hidden_states) * y_mask.unsqueeze(1)

        return hidden_states[:, :, 1:]