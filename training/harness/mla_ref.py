# -*- coding: utf-8 -*-
"""DeepSeek MLA（Multi-head Latent Attention）的**最小可验证实现**（B5 第一步）。

MLA 的核心不是"多头注意力"，而是**KV 走隐空间压缩**：
    c_kv = W_dkv(x)                    # [N, kv_lora_rank]  ← 真正的 KV 只有这一份
    k_nope = W_uk(c_kv)                # 每头 [qk_nope_head_dim]
    k_rope = W_kr(c_kv)                # **所有头共享** [qk_rope_head_dim]
    v      = W_uv(c_kv)                # 每头 [v_head_dim]
    q = W_uq(W_dq(x)) 或 W_q(x)        # 每头 [qk_nope_head_dim + qk_rope_head_dim]
分数用 q·k 拼接（nope 部分逐头，rope 部分共享），输出再经 W_o。

**本文件的范围（诚实边界）**：
- 实现了：隐空间压缩、nope/rope 拆分、共享 k_rope、Q 的低秩投影（可选）、输出投影
- **未实现**：RoPE 本身（本步先验证结构）、完整的 DeepSeek MoE 路由、吸收矩阵优化
本步判据：与 PyTorch 参考实现**同权重同输入**下 logits 相对误差 < 1e-4。
"""
import numpy as np

import tgai_go.train as T


class MLA:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = {k: T.Parameter(v) for k, v in weights.items()}

    def parameters(self):
        return list(self.w.values())

    def _lin(self, x, name):
        return x @ self.w[name + ".weight"].T

    def forward(self, ids, tgt=None):
        c = self.cfg
        D, NH = c["d"], c["nh"]
        NOPE, ROPE, VH = c["qk_nope"], c["qk_rope"], c["v_head"]
        KL = c["kv_lora"]
        ids = np.asarray(ids, dtype=np.int64)
        B, S = int(ids.shape[0]), int(ids.shape[1])

        emb = T.nn.Embedding(c["V"], D)
        emb.weight = self.w["embed.weight"]
        x = emb(ids).reshape(B * S, D)

        # ── KV 走隐空间：真正的 KV 只有 c_kv 一份 ──
        ckv = self._lin(x, "kv_dkv")                              # [N, KL]
        k_nope = self._lin(ckv, "kv_uk").reshape(B, S, NH, NOPE)
        k_rope = self._lin(ckv, "kv_kr")                          # [N, ROPE]，**全头共享**
        v = self._lin(ckv, "kv_uv").reshape(B, S, NH, VH)

        # ── Q（可低秩）──
        if c.get("q_lora"):
            qh = self._lin(self._lin(x, "q_dq"), "q_uq")
        else:
            qh = self._lin(x, "q_proj")
        q = qh.reshape(B, S, NH, NOPE + ROPE)

        # 统一成 [B*NH, S, ·]
        qn = q[..., :NOPE].permute(0, 2, 1, 3).reshape(B * NH, S, NOPE)
        qr = q[..., NOPE:].permute(0, 2, 1, 3).reshape(B * NH, S, ROPE)
        kn = k_nope.permute(0, 2, 1, 3).reshape(B * NH, S, NOPE)
        vh = v.permute(0, 2, 1, 3).reshape(B * NH, S, VH)

        # k_rope 全头共享 ⇒ 按头平铺（用 numpy 落地，本步只验前向结构）
        kr_np = np.asarray(k_rope.to_numpy(), np.float32).reshape(B, S, ROPE)
        kr = T.Tensor(np.tile(kr_np, (1, NH, 1)).reshape(B * NH, S, ROPE))

        qcat = T.cat([qn, qr], dim=-1)      # [B*NH, S, NOPE+ROPE]
        kcat = T.cat([kn, kr], dim=-1)
        o = T.functional.scaled_dot_product_attention(
            qcat, kcat, vh, causal=True, scale=1.0 / np.sqrt(float(NOPE + ROPE)))
        o = o.reshape(B, NH, S, VH).permute(0, 2, 1, 3).reshape(B * S, NH * VH)
        out = self._lin(o, "o_proj")
        logits = out @ self.w["lm_head.weight"].T
        return logits.reshape(B, S, c["V"])


def make_cfg(d=64, nh=4, kv_lora=32, qk_nope=16, qk_rope=8, v_head=16,
             V=64, seq=16, q_lora=False, seed=0):
    g = np.random.RandomState(seed)
    s = 0.02

    def rn(*sh):
        return (g.randn(*sh) * s).astype(np.float32)

    D = d
    w = {"embed.weight": rn(V, D), "lm_head.weight": rn(V, D),
         "kv_dkv.weight": rn(kv_lora, D),               # [KL, D]  压到隐空间
         "kv_uk.weight": rn(nh * qk_nope, kv_lora),     # [NH*NOPE, KL]
         "kv_kr.weight": rn(qk_rope, kv_lora),          # [ROPE, KL]  全头共享
         "kv_uv.weight": rn(nh * v_head, kv_lora),      # [NH*VH, KL]
         "q_proj.weight": rn(nh * (qk_nope + qk_rope), D),
         "o_proj.weight": rn(D, nh * v_head)}
    if q_lora:
        w["q_dq.weight"] = rn(48, D)
        w["q_uq.weight"] = rn(nh * (qk_nope + qk_rope), 48)
    cfg = dict(d=D, nh=nh, kv_lora=kv_lora, qk_nope=qk_nope, qk_rope=qk_rope,
               v_head=v_head, V=V, seq=seq, q_lora=bool(q_lora), weights=w)
    return cfg
