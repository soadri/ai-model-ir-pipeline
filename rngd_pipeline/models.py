"""
models.py — RNGD Lowering 파이프라인 공통 모델 정의
Llama-3.1 8B Decoder Layer (GQA + RoPE) 구조.
모든 검증/분석 도구가 이 파일의 모델을 공유합니다.
"""

import torch
import torch.nn as nn


class LlamaRotaryEmbedding(nn.Module):
    """RoPE — 사전 계산 방식 (torch_mlir FX 경로 호환)"""
    def __init__(self, dim, seq_len=128, base=500000.0):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        freqs = torch.outer(torch.arange(seq_len).float(), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0))
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0))

    def forward(self, seq_len: int):
        return self.cos_cached[:, :seq_len, :], self.sin_cached[:, :seq_len, :]


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class LlamaAttention(nn.Module):
    """Multi-Head Attention (GQA 지원, RoPE 포함)"""
    def __init__(self, dim, n_heads, n_kv_heads, head_dim, seq_len=128, always_repeat_kv=False):
        super().__init__()
        # always_repeat_kv: n_rep=1이어도 repeat_interleave를 수행 (대시보드 기본 모델의 기존 IR 구성 유지용)
        self.always_repeat_kv = always_repeat_kv
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads
        self.head_dim = head_dim
        self.n_rep = n_heads // n_kv_heads
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_kv_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)
        self.rotary_emb = LlamaRotaryEmbedding(head_dim, seq_len=seq_len)
        self.scale = head_dim ** -0.5

    def forward(self, x):
        B, S, D = x.shape
        q = self.q_proj(x).view(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, S, self.n_kv_heads, self.head_dim).transpose(1, 2)
        cos, sin = self.rotary_emb(S)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        if self.n_rep > 1 or self.always_repeat_kv:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.o_proj(out)


class LlamaRMSNorm(nn.Module):
    """RMSNorm"""
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight


class LlamaMLP(nn.Module):
    """FFN (SwiGLU)"""
    def __init__(self, dim, hidden):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden, bias=False)
        self.up_proj = nn.Linear(dim, hidden, bias=False)
        self.down_proj = nn.Linear(hidden, dim, bias=False)

    def forward(self, x):
        return self.down_proj(torch.nn.functional.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaDecoderLayer(nn.Module):
    """Llama-3.1 Decoder Layer (RMSNorm + Attention + FFN + Residual)"""
    def __init__(self, dim, hidden, n_heads, n_kv_heads, head_dim, seq_len=128, rms_eps=1e-5,
                 always_repeat_kv=False):
        super().__init__()
        self.input_layernorm = LlamaRMSNorm(dim, rms_eps)
        self.self_attn = LlamaAttention(dim, n_heads, n_kv_heads, head_dim, seq_len, always_repeat_kv)
        self.post_attention_layernorm = LlamaRMSNorm(dim, rms_eps)
        self.mlp = LlamaMLP(dim, hidden)

    def forward(self, x):
        x = x + self.self_attn(self.input_layernorm(x))
        x = x + self.mlp(self.post_attention_layernorm(x))
        return x


# 검증 프로파일에서 모델을 생성하는 유틸리티
def create_model_from_profile(profile):
    """프로파일 dict로 LlamaDecoderLayer 생성"""
    model = LlamaDecoderLayer(
        dim=profile["dim"],
        hidden=profile["hidden"],
        n_heads=profile["n_heads"],
        n_kv_heads=profile["n_kv_heads"],
        head_dim=profile["head_dim"],
        seq_len=profile["seq_len"],
        rms_eps=profile.get("rms_eps", 1e-5),
    )
    model.eval()
    return model
