# -*- coding: utf-8 -*-
"""GPT-2 的薄定义（B2 的语义基准）。

与 llama 的差异（这正是"一份核心、多套架构"要覆盖的点）：
- **LayerNorm**（可学习 gamma/beta，带偏置）而不是 RMSNorm
- **GELU（tanh 近似，GPT-2 的 gelu_new）** 而不是 SiLU
- **学习式位置编码**（wpe）而不是 RoPE
- **所有线性层带 bias**
- **因果掩码但用 F.scaled_dot_product_attention 的 is_causal**
- 头数 == kv 头数（MHA，无 GQA）
- lm_head 与 wte **不共享**（按目标要求 untied；真实 GPT-2 是 tied，用 tie=True 可切换）
"""
import numpy as np

import tgai_go.train as T


def _sigmoid(x):
    """sigmoid(x) = 1/(1+exp(-x))，用已实现的 exp 组合。"""
    return (( -x).exp() + 1.0) ** -1.0


def gelu_tanh(x):
    """GPT-2 的 gelu_new：0.5x(1+tanh(sqrt(2/pi)(x+0.044715x^3)))。

    tanh(z) = 2*sigmoid(2z) - 1，避免依赖未实现的 tanh。
    """
    import math

    c = math.sqrt(2.0 / math.pi)
    x3 = x * x * x
    inner = (x + x3 * 0.044715) * c
    t = _sigmoid(inner * 2.0) * 2.0 - 1.0
    return x * (t + 1.0) * 0.5


def _layernorm(x, gamma, beta, eps=1e-5):
    """[N, D] 的 LayerNorm。CUDA 上 mean/var 尚未实现，届时需换成 nn.LayerNorm 级 kernel。"""
    mu = x.mean(axis=-1, keepdims=True)
    xc = x - mu
    var = (xc * xc).mean(axis=-1, keepdims=True)
    inv = (var + T.Tensor(np.array([eps], np.float32))) ** -0.5
    return xc * inv * gamma + beta


class GPT2:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = {k: T.Parameter(v) for k, v in weights.items()}

    def parameters(self):
        return list(self.w.values())

    def _ln(self, x, name):
        return _layernorm(x, self.w[name + ".weight"], self.w[name + ".bias"])

    def _linear(self, x, name, bias=True):
        out = x @ self.w[name + ".weight"].T
        if bias:
            out = out + self.w[name + ".bias"]
        return out

    def forward(self, ids):
        c = self.cfg
        D, NH, L, S = c["d"], c["nh"], c["L"], int(ids.shape[1])
        B = int(ids.shape[0])
        DH = D // NH

        emb = T.nn.Embedding(c["V"], D)
        emb.weight = self.w["wte.weight"]
        tok = emb(np.asarray(ids, dtype=np.int64)).reshape(B, S, D)
        pos_idx = np.arange(S, dtype=np.int64)
        pos = T.nn.Embedding(S, D)
        pos.weight = self.w["wpe.weight"]
        h = tok + pos(pos_idx).reshape(1, S, D)

        mask = None
        for i in range(L):
            p = "blocks.%d." % i
            n = self._ln(h, p + "ln_1")
            qkv = self._linear(n.reshape(B * S, D), p + "attn.qkv")   # [B*S, 3D]
            q3 = qkv.reshape(B, S, 3, NH, DH)
            # 用切片取 q/k/v（CUDA 已支持基本切片）
            q = q3[:, :, 0].permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            k = q3[:, :, 1].permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            v = q3[:, :, 2].permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            o = T.functional.scaled_dot_product_attention(q, k, v, causal=True)
            o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
            h = h + self._linear(o, p + "attn.proj").reshape(B, S, D)

            n2 = self._ln(h, p + "ln_2")
            mid = gelu_tanh(self._linear(n2.reshape(B * S, D), p + "mlp.fc"))
            h = h + self._linear(mid, p + "mlp.proj").reshape(B, S, D)

        hn = self._ln(h, "ln_f")
        logits = hn.reshape(B * S, D) @ self.w["lm_head"].T
        if c.get("tie"):
            logits = hn.reshape(B * S, D) @ self.w["wte.weight"].T
        return logits


def init_weights(cfg, seed=0):
    g = np.random.RandomState(seed)
    D, L, FF, V, S = cfg["d"], cfg["L"], cfg["ff"], cfg["V"], cfg["seq"]
    s = 0.02

    def rn(*shape):
        return (g.randn(*shape) * s).astype(np.float32)

    w = {"wte.weight": rn(V, D), "wpe.weight": rn(S, D),
         "ln_f.weight": np.ones(D, np.float32), "ln_f.bias": np.zeros(D, np.float32),
         "lm_head": rn(V, D)}
    for i in range(L):
        p = "blocks.%d." % i
        for nm in ("ln_1", "ln_2"):
            w[p + nm + ".weight"] = np.ones(D, np.float32)
            w[p + nm + ".bias"] = np.zeros(D, np.float32)
        w[p + "attn.qkv.weight"] = rn(3 * D, D)
        w[p + "attn.qkv.bias"] = np.zeros(3 * D, np.float32)
        w[p + "attn.proj.weight"] = rn(D, D)
        w[p + "attn.proj.bias"] = np.zeros(D, np.float32)
        w[p + "mlp.fc.weight"] = rn(FF, D)
        w[p + "mlp.fc.bias"] = np.zeros(FF, np.float32)
        w[p + "mlp.proj.weight"] = rn(D, FF)
        w[p + "mlp.proj.bias"] = np.zeros(D, np.float32)
    return w


def make_cfg(d=32, L=2, nh=4, ff=64, V=64, seq=16, tie=False, seed=0):
    cfg = dict(d=d, L=L, nh=nh, ff=ff, V=V, seq=seq, tie=bool(tie), seed=seed)
    cfg["weights"] = init_weights(cfg, seed)
    return cfg
