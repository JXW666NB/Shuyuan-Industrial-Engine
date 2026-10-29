# -*- coding: utf-8 -*-
"""llama 的薄定义（B1 的语义基准）。

目的：先用**现有算子**把 llama 的数学跑通并锁死数值，给 C 移植一个绿色参照。
结构（与官方 llama 一致）：RMSNorm → RoPE → GQA causal attn → 残差
                                 → RMSNorm → SwiGLU FFN → 残差
                                 → final RMSNorm → 输出头（untied）

RoPE 用**组合式**实现（slicing + cat + mul/add），只求正确，性能不重要；
C 版本应改用 rope kernel。
"""
import numpy as np

import tgai_go.train as T


def rope_tables(seq, head_dim, theta=10000.0, base_dtype=np.float32):
    """返回 cos/sin，形状 [seq, head_dim]（前后两半相同，配合 rotate-half）。"""
    half = head_dim // 2
    inv = 1.0 / (theta ** (np.arange(0, half, dtype=np.float64) * 2.0 / head_dim))
    pos = np.arange(seq, dtype=np.float64)[:, None]
    ang = pos * inv[None, :]                       # [seq, half]
    cos = np.cos(ang).astype(base_dtype)
    sin = np.sin(ang).astype(base_dtype)
    return np.concatenate([cos, cos], axis=1), np.concatenate([sin, sin], axis=1)


def _rotate_perm(head_dim):
    """返回 R[dh,dh]，使 x @ R == [-x2, x1]（前半取负换到后半）。

    用矩阵乘而不是切片：R 的元素只有 0 / ±1，浮点上精确。
    切片已经能在 CUDA 上跑，但这里保持原来的 matmul，避免给 RoPE 换一条数值路径。
    """
    half = head_dim // 2
    R = np.zeros((head_dim, head_dim), np.float32)
    for j in range(half):
        R[j + half, j] = -1.0      # 输出的第 j 列 = -x[j+half]
    for j in range(half, head_dim):
        R[j - half, j] = 1.0       # 输出的第 j 列 = x[j-half]
    return R


def _rope(x, cos_np, sin_np, R_np):
    """x: [B*H, S, Dh]；cos/sin: [S, Dh]（会平铺）；R: [Dh, Dh]。"""
    bh, s, dh = (int(v) for v in x.shape)
    c = np.tile(np.asarray(cos_np), (bh, 1, 1))[:, :s, :dh]
    sn = np.tile(np.asarray(sin_np), (bh, 1, 1))[:, :s, :dh]
    rot = x @ T.Tensor(np.ascontiguousarray(R_np, dtype=np.float32))
    return x * T.Tensor(np.ascontiguousarray(c, dtype=np.float32)) + \
        rot * T.Tensor(np.ascontiguousarray(sn, dtype=np.float32))


def _repeat_kv(x, n_rep):
    """[B*Hkv, S, Dh] -> [B*Hkv*n_rep, S, Dh]（每个 kv head 连续重复 n_rep 次）。

    走 repeat_interleave，不再用 x[i:i+1] 拼 cat：CUDA 上 cat 还没实现。
    """
    if int(n_rep) == 1:
        return x
    return T.repeat_interleave(x, int(n_rep), dim=0)


class Llama:
    """最小 llama：参数用 numpy 给出的 dict，便于与 PyTorch 侧逐位共享。"""

    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = {k: T.Parameter(v) for k, v in weights.items()}
        self._cos = np.asarray(cfg["cos"], dtype=np.float32)
        self._sin = np.asarray(cfg["sin"], dtype=np.float32)
        self._R = _rotate_perm(int(cfg["d"] // cfg["nh"]))

    def parameters(self):
        return list(self.w.values())

    def _rms(self, x, name, eps=1e-6):
        # 必须用 nn.RMSNorm（已接 CUDA kernel）；手写 x*x.mean(...) 在 CUDA 上会因缺 mean 而失败。
        mod = T.nn.RMSNorm(int(x.shape[-1]), eps=eps)
        mod.weight = self.w[name + ".weight"]
        return mod(x)

    def forward(self, ids):
        c = self.cfg
        D, NH, NKV = c["d"], c["nh"], c["n_kv"]
        DH = D // NH
        n_rep = NH // NKV
        B, S = int(ids.shape[0]), int(ids.shape[1])

        # 用 nn.Embedding 的 CUDA 路径（row-gather 已有 kernel）；权重与外部共享同一 Parameter。
        # __getitem__ 只做整数/切片，不做数组下标，所以不能写 tok_emb.weight[ids]。
        emb = T.nn.Embedding(c["V"], D)
        emb.weight = self.w["tok_emb.weight"]
        h = emb(np.asarray(ids, dtype=np.int64)).reshape(B, S, D)
        for i in range(c["L"]):
            p = "blocks.%d." % i
            n = self._rms(h, p + "attn_norm")
            q = self.w[p + "attn.q.weight"]
            k = self.w[p + "attn.k.weight"]
            v = self.w[p + "attn.v.weight"]

            nf = n.reshape(B * S, D)
            qf = nf @ q.T
            kf = nf @ k.T
            vf = nf @ v.T
            if c.get("qkv_bias"):     # Qwen2 的差异就在这三行 bias
                qf = qf + self.w[p + "attn.q.bias"]
                kf = kf + self.w[p + "attn.k.bias"]
                vf = vf + self.w[p + "attn.v.bias"]
            qh = qf.reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
            kh = kf.reshape(B, S, NKV, DH).permute(0, 2, 1, 3).reshape(B * NKV, S, DH)
            vh = vf.reshape(B, S, NKV, DH).permute(0, 2, 1, 3).reshape(B * NKV, S, DH)

            qh = _rope(qh, self._cos, self._sin, self._R)
            kh = _rope(kh, self._cos, self._sin, self._R)
            kh = _repeat_kv(kh, n_rep)
            vh = _repeat_kv(vh, n_rep)

            o = T.functional.scaled_dot_product_attention(qh, kh, vh, causal=True)
            o = o.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
            h = h + (o @ self.w[p + "attn.o.weight"].T).reshape(B, S, D)

            n2 = self._rms(h, p + "ffn_norm")
            gu = T.functional.silu(n2.reshape(B * S, D) @ self.w[p + "ffn.gate.weight"].T) * \
                (n2.reshape(B * S, D) @ self.w[p + "ffn.up.weight"].T)
            h = h + (gu @ self.w[p + "ffn.down.weight"].T).reshape(B, S, D)

        hn = self._rms(h, "final_norm")
        return hn.reshape(B * S, D) @ self.w["lm_head.weight"].T


def init_weights(cfg, seed=0):
    """与 PyTorch 侧共享的初始权重（numpy，float32）。"""
    g = np.random.RandomState(seed)
    D, L, FF, V, NKV = cfg["d"], cfg["L"], cfg["ff"], cfg["V"], cfg["n_kv"]
    s = 1.0 / np.sqrt(D)

    def rn(*shape, scale=s):
        return (g.randn(*shape) * scale).astype(np.float32)

    w = {"tok_emb.weight": rn(V, D), "lm_head.weight": rn(V, D),
         "final_norm.weight": np.ones(D, np.float32)}
    for i in range(L):
        p = "blocks.%d." % i
        w[p + "attn_norm.weight"] = np.ones(D, np.float32)
        w[p + "ffn_norm.weight"] = np.ones(D, np.float32)
        if cfg.get("qkv_bias"):
            # 必须给非零值：全零的 bias 让 Qwen2 等价于 llama，判分台就白跑了
            w[p + "attn.q.bias"] = rn(D)
            w[p + "attn.k.bias"] = rn(NKV * (D // cfg["nh"]))
            w[p + "attn.v.bias"] = rn(NKV * (D // cfg["nh"]))
        w[p + "attn.q.weight"] = rn(D, D)
        w[p + "attn.k.weight"] = rn(NKV * (D // cfg["nh"]), D)
        w[p + "attn.v.weight"] = rn(NKV * (D // cfg["nh"]), D)
        w[p + "attn.o.weight"] = rn(D, D)
        w[p + "ffn.gate.weight"] = rn(FF, D)
        w[p + "ffn.up.weight"] = rn(FF, D)
        w[p + "ffn.down.weight"] = rn(D, FF)
    return w


def make_cfg(d=32, L=2, nh=4, n_kv=2, ff=64, V=64, seq=16, theta=10000.0, seed=0,
             qkv_bias=False):
    cos, sin = rope_tables(seq, d // nh, theta)
    cfg = dict(d=d, L=L, nh=nh, n_kv=n_kv, ff=ff, V=V, seq=seq, theta=theta,
               cos=cos, sin=sin, seed=seed, qkv_bias=bool(qkv_bias))
    cfg["weights"] = init_weights(cfg, seed)
    return cfg
