"""Gemma 3 계열 소형 디코더 + MTP(미래 16토큰 예측) 단일 모델.

구성 요소:
  - RMSNorm pre+post 샌드위치 정규화, bias 없음
  - GQA + QK-norm, RoPE (local/global theta 분리)
  - sliding-window local : global 어텐션 5:1 교차 배치
  - GeGLU MLP, 임베딩 sqrt(d) 스케일링, 입출력 임베딩 tying
  - MTP: 공유 경량 블록 + offset별 임베딩 -> tied unembedding 으로
    offset 1..mtp_n 의 미래 토큰을 병렬 예측 (학습 전용, 추론 시 미사용)

forward(input_ids, loss_mask) 는 내부에서 라벨을 시프트해
main CE / MTP CE 를 계산한다. loss_mask[t]==1 인 위치의 토큰만 학습 대상.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 10240
    n_layers: int = 24
    d_model: int = 768
    n_heads: int = 12
    n_kv_heads: int = 4
    head_dim: int = 64
    ffn_hidden: int = 2048
    max_seq_len: int = 2048
    sliding_window: int = 512
    global_every: int = 6  # 6개 층마다 1개 global (5 local : 1 global)
    rope_theta_local: float = 10_000.0
    rope_theta_global: float = 1_000_000.0
    rms_eps: float = 1e-6
    mtp_n: int = 16
    mtp_weight: float = 0.2
    mtp_ffn_hidden: int = 1024
    pad_id: int = 0
    bos_id: int = 1
    eos_id: int = 2

    def is_global_layer(self, idx: int) -> bool:
        return (idx + 1) % self.global_every == 0

    def to_dict(self) -> dict:
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.float()
        x = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return (x * self.weight.float()).to(dtype)


def build_rope_cache(seq_len: int, head_dim: int, theta: float, device, dtype=torch.float32):
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device, dtype=torch.float32) / head_dim))
    t = torch.arange(seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos().to(dtype), freqs.sin().to(dtype)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, H, T, D), cos/sin: (T, D/2)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    cos, sin = cos[None, None, :, :], sin[None, None, :, :]
    out = torch.empty_like(x)
    out[..., 0::2] = x1 * cos - x2 * sin
    out[..., 1::2] = x1 * sin + x2 * cos
    return out


class Attention(nn.Module):
    def __init__(self, cfg: ModelConfig, is_global: bool):
        super().__init__()
        self.cfg = cfg
        self.is_global = is_global
        self.n_heads = cfg.n_heads
        self.n_kv_heads = cfg.n_kv_heads
        self.head_dim = cfg.head_dim
        q_dim = cfg.n_heads * cfg.head_dim
        kv_dim = cfg.n_kv_heads * cfg.head_dim
        self.wq = nn.Linear(cfg.d_model, q_dim, bias=False)
        self.wk = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wv = nn.Linear(cfg.d_model, kv_dim, bias=False)
        self.wo = nn.Linear(q_dim, cfg.d_model, bias=False)
        self.q_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)
        self.k_norm = RMSNorm(cfg.head_dim, cfg.rms_eps)

    def forward(self, x, cos, sin, attn_mask, kv_cache=None):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        q, k = self.q_norm(q), self.k_norm(k)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v)

        k = k.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)
        v = v.repeat_interleave(self.n_heads // self.n_kv_heads, dim=1)

        out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        out = out.transpose(1, 2).reshape(B, T, -1)
        return self.wo(out)


class GeGLU(nn.Module):
    def __init__(self, d_model: int, hidden: int):
        super().__init__()
        self.gate = nn.Linear(d_model, hidden, bias=False)
        self.up = nn.Linear(d_model, hidden, bias=False)
        self.down = nn.Linear(hidden, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.gelu(self.gate(x), approximate="tanh") * self.up(x))


class Block(nn.Module):
    def __init__(self, cfg: ModelConfig, layer_idx: int):
        super().__init__()
        self.is_global = cfg.is_global_layer(layer_idx)
        self.attn = Attention(cfg, self.is_global)
        self.mlp = GeGLU(cfg.d_model, cfg.ffn_hidden)
        self.pre_attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.post_attn_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.pre_mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.post_mlp_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, x, rope, masks, kv_cache=None):
        cos, sin = rope["global" if self.is_global else "local"]
        mask = masks["global" if self.is_global else "local"]
        x = x + self.post_attn_norm(self.attn(self.pre_attn_norm(x), cos, sin, mask, kv_cache))
        x = x + self.post_mlp_norm(self.mlp(self.pre_mlp_norm(x)))
        return x


class MTPHead(nn.Module):
    """공유 GeGLU 블록 + offset별 임베딩. 로짓은 tied unembedding 으로 계산."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.offset_emb = nn.Parameter(torch.zeros(cfg.mtp_n, cfg.d_model))
        self.mlp = GeGLU(cfg.d_model, cfg.mtp_ffn_hidden)
        self.out_norm = RMSNorm(cfg.d_model, cfg.rms_eps)

    def forward(self, h: torch.Tensor, offset_idx: int) -> torch.Tensor:
        # h: (B, T, D) -> offset_idx(0-based, 예측 offset = offset_idx+1) 의 hidden
        z = self.norm(h) + self.offset_emb[offset_idx]
        return self.out_norm(z + self.mlp(z))


class KVCache:
    def __init__(self):
        self.k = self.v = None

    def update(self, k, v):
        if self.k is None:
            self.k, self.v = k, v
        else:
            self.k = torch.cat([self.k, k], dim=2)
            self.v = torch.cat([self.v, v], dim=2)
        return self.k, self.v


class KoreanSLLM(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.blocks = nn.ModuleList(Block(cfg, i) for i in range(cfg.n_layers))
        self.final_norm = RMSNorm(cfg.d_model, cfg.rms_eps)
        self.mtp_head = MTPHead(cfg)
        self.apply(self._init_weights)
        for name, p in self.named_parameters():
            if name.endswith(("wo.weight", "down.weight")):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * cfg.n_layers))

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def _rope(self, T: int, offset: int, device, dtype):
        rope = {}
        for kind, theta in (("local", self.cfg.rope_theta_local), ("global", self.cfg.rope_theta_global)):
            cos, sin = build_rope_cache(offset + T, self.cfg.head_dim, theta, device, dtype)
            rope[kind] = (cos[offset:], sin[offset:])
        return rope

    def _masks(self, q_len: int, kv_len: int, device):
        q_pos = torch.arange(kv_len - q_len, kv_len, device=device)[:, None]
        k_pos = torch.arange(kv_len, device=device)[None, :]
        causal = k_pos <= q_pos
        local = causal & (q_pos - k_pos < self.cfg.sliding_window)
        return {"global": causal, "local": local}

    def _trunk(self, input_ids, kv_caches=None, pos_offset: int = 0):
        B, T = input_ids.shape
        x = self.embed(input_ids) * math.sqrt(self.cfg.d_model)
        rope = self._rope(T, pos_offset, input_ids.device, x.dtype)
        masks = self._masks(T, pos_offset + T, input_ids.device)
        for i, block in enumerate(self.blocks):
            x = block(x, rope, masks, kv_caches[i] if kv_caches else None)
        return self.final_norm(x)

    def logits_from_hidden(self, h: torch.Tensor) -> torch.Tensor:
        return h @ self.embed.weight.t()

    def forward(self, input_ids: torch.Tensor, loss_mask: torch.Tensor | None = None):
        """loss_mask 가 주어지면 {loss, main_loss, mtp_loss} 반환, 없으면 로짓 반환."""
        h = self._trunk(input_ids)
        if loss_mask is None:
            return self.logits_from_hidden(h)

        B, T = input_ids.shape
        # main: 위치 t 에서 t+1 예측
        main_logits = self.logits_from_hidden(h[:, :-1])
        main_targets = input_ids[:, 1:].masked_fill(loss_mask[:, 1:] == 0, -100)
        main_loss = F.cross_entropy(
            main_logits.reshape(-1, self.cfg.vocab_size).float(),
            main_targets.reshape(-1), ignore_index=-100,
        )

        # MTP: 위치 t 에서 t+1+k 예측 (k = 1..mtp_n). offset 이 멀수록 선형 감쇠 가중.
        mtp_losses = []
        weights = []
        for k in range(1, self.cfg.mtp_n + 1):
            if T <= k + 1:
                break
            hk = self.mtp_head(h[:, : T - 1 - k], offset_idx=k - 1)
            logits_k = self.logits_from_hidden(hk)
            targets_k = input_ids[:, 1 + k:].masked_fill(loss_mask[:, 1 + k:] == 0, -100)
            loss_k = F.cross_entropy(
                logits_k.reshape(-1, self.cfg.vocab_size).float(),
                targets_k.reshape(-1), ignore_index=-100,
            )
            if torch.isfinite(loss_k):  # 해당 offset 에 유효 타깃이 없으면 제외
                w = 1.0 - 0.5 * (k - 1) / max(self.cfg.mtp_n - 1, 1)
                mtp_losses.append(loss_k * w)
                weights.append(w)

        mtp_loss = torch.stack(mtp_losses).sum() / sum(weights) if mtp_losses else main_loss.new_zeros(())
        loss = main_loss + self.cfg.mtp_weight * mtp_loss
        return {"loss": loss, "main_loss": main_loss.detach(), "mtp_loss": mtp_loss.detach()}

    @torch.inference_mode()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 256,
                 temperature: float = 0.8, top_p: float = 0.95) -> torch.Tensor:
        self.eval()
        kv_caches = [KVCache() for _ in self.blocks]
        ids = input_ids
        h = self._trunk(ids, kv_caches, pos_offset=0)
        for _ in range(max_new_tokens):
            logits = self.logits_from_hidden(h[:, -1]).float()
            if temperature <= 0:
                next_id = logits.argmax(-1, keepdim=True)
            else:
                probs = F.softmax(logits / temperature, dim=-1)
                sorted_probs, sorted_idx = probs.sort(descending=True)
                cum = sorted_probs.cumsum(-1)
                sorted_probs[cum - sorted_probs > top_p] = 0.0
                sorted_probs /= sorted_probs.sum(-1, keepdim=True)
                next_id = sorted_idx.gather(-1, torch.multinomial(sorted_probs, 1))
            ids = torch.cat([ids, next_id], dim=1)
            if (next_id == self.cfg.eos_id).all() or ids.shape[1] >= self.cfg.max_seq_len:
                break
            h = self._trunk(next_id, kv_caches, pos_offset=ids.shape[1] - 1)
        return ids
