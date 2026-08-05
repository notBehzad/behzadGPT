import torch
import torch.nn as nn
import torch.nn.functional as F

CONFIG = {
    # Architecture
    "vocab_size": 8192,
    "d_model": 448,
    "d_ff": 1280,         # 2.6667x the d_model but closest 256 multiple
    "context_length": 512,
    "d_head": 64,
    "layer_count": 9,
}


def precompute_rope_freqs(d_head, seq_len):
    i = torch.arange(0, d_head // 2, dtype=torch.float32)
    theta = 10000.0 ** (-2 * i / d_head)
    m = torch.arange(0, seq_len, dtype=torch.float32)

    freqs = torch.outer(m, theta)
    freqs = torch.cat([freqs, freqs], dim=-1)

    return torch.sin(freqs), torch.cos(freqs)


def apply_RoPE(x, sin, cos):
    sin = sin.unsqueeze(0).unsqueeze(0)  # (1, 1, seq_len, d_head)
    cos = cos.unsqueeze(0).unsqueeze(0)

    half_d = x.shape[-1] // 2
    rotated_x = torch.cat([-x[..., half_d:], x[..., :half_d]], dim=-1)
    return (x * cos) + (rotated_x * sin)


class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, d_model, d_head, context_length):
        super().__init__()
        if d_model % d_head != 0:
            raise ValueError("d_model and d_head are not divisible.")
        self.num_heads = d_model // d_head
        self.d_head = d_head
        self.d_model = d_model

        self.Q = nn.Linear(d_model, d_model, bias=False)
        self.K = nn.Linear(d_model, d_model, bias=False)
        self.V = nn.Linear(d_model, d_model, bias=False)
        self.O = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x, sin, cos):
        B, S, _ = x.shape

        Q = apply_RoPE(self.Q(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        K = apply_RoPE(self.K(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2), sin[:S], cos[:S])
        V = self.V(x).view(B, S, self.num_heads, self.d_head).transpose(1, 2)

        logits = (Q @ K.transpose(-1, -2)) / (self.d_head ** 0.5)

        mask = torch.triu(torch.ones(S, S, device=x.device, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(mask, float("-inf"))

        attn_weights = F.softmax(logits, dim=-1)
        heads_out = attn_weights @ V

        heads_out = heads_out.transpose(1, 2).contiguous().view(B, S, self.d_model)
        return self.O(heads_out)


class SwiGELU_FFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, d_head, seq_len, d_ff):
        super().__init__()
        self.attention = MultiHeadAttentionBlock(d_model, d_head, seq_len)
        self.ffn = SwiGELU_FFN(d_model, d_ff)
        self.norm1 = nn.RMSNorm(d_model)
        self.norm2 = nn.RMSNorm(d_model)
        self.attention.O.special_init = True
        self.ffn.down.special_init = True

    def forward(self, x, sin, cos):
        x = x + self.attention(self.norm1(x), sin, cos)
        return x + self.ffn(self.norm2(x))


class behzadGPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = nn.Embedding(config["vocab_size"], config["d_model"])
        sin, cos = precompute_rope_freqs(config["d_head"], config["context_length"])
        self.register_buffer("sin", sin, persistent=False)
        self.register_buffer("cos", cos, persistent=False)

        self.layers = nn.ModuleList([
            TransformerBlock(
                config["d_model"],
                config["d_head"],
                config["context_length"],
                config["d_ff"]
            ) for _ in range(config["layer_count"])
        ])

    def forward(self, x):
        x = self.embeddings(x)
        for layer in self.layers:
            x = layer(x, self.sin, self.cos)
        return torch.matmul(x, self.embeddings.weight.T)
