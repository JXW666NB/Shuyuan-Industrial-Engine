# -*- coding: utf-8 -*-
"""TGAI V3 的非 CSA 部分 —— **只做前向**（B4 第一阶段），与官方 oracle 对拍。

为什么先只做前向：MoE 的按专家 gather/scatter 需要 fancy indexing，当前张量层在 CUDA 上还不完整；
前向可以用 numpy 做组装（不需求导），先把**数值对齐**拿下来，反向留到下一阶段。

源码事实（docs/v3_architecture_facts.md）：
- 每层两个 BlockAttnRes（attn 前 / mlp 前）：对 blocks+partial 做 softmax 伪查询注意力
- attention：q/k/v proj → **QK-Norm（在 RoPE 前）** → RoPE → GQA repeat → causal SDPA → o_proj
- MoE：sigmoid 路由 + expert_bias 只影响选择 + 选中集内归一化 + SiTU-GLU 专家
        → topk 求和 → post_norm → 共享专家无门控相加
- 输出头与词表共享（tie_embeddings=True）
"""
import numpy as np

import tgai_go.train as T

B1, B2 = 4.0, 25.0


def _sigmoid(x):
    return ((-x).exp() + 1.0) ** -1.0


def _tanh(x):
    return _sigmoid(x * 2.0) * 2.0 - 1.0


def situ_glu(g, u):
    return (_tanh(g * (1.0 / B1)) * B1) * _sigmoid(g) * (_tanh(u * (1.0 / B2)) * B2)


def _rh(x):
    """V3 的 rotate_half 是**交错式**：(x1,x2,x3,x4) → (-x2,x1,-x4,x3)。

    注意与 llama/GPT-2 的"半切"不同 —— 这是我对拍时踩的第一个坑。
    官方源码：r[..., ::2] = -x[..., 1::2]；r[..., 1::2] = x[..., ::2]
    """
    a = np.asarray(x.to_numpy(), dtype=np.float32)
    r = np.empty_like(a)
    r[..., ::2] = -a[..., 1::2]
    r[..., 1::2] = a[..., ::2]
    return T.Tensor(r)


class V3Forward:
    def __init__(self, cfg, weights):
        self.cfg = cfg
        self.w = {k: T.Parameter(v) for k, v in weights.items()}

    def _rms(self, x, name, eps=1e-6):
        mod = T.nn.RMSNorm(int(x.shape[-1]), eps=eps)
        mod.weight = self.w[name]
        return mod(x)

    def _lin(self, x, name):
        return x @ self.w[name].T

    def _attn_res(self, blocks, partial, prefix, i):
        if not blocks:
            return partial
        B, S, D = (int(v) for v in partial.shape)
        V = [b.reshape(B * S, D) for b in blocks] + [partial.reshape(B * S, D)]
        Ks = [self._rms(t, "%s.%d.norm.weight" % (prefix, i)) for t in V]
        wv = self.w["%s.%d.proj.weight" % (prefix, i)].reshape((D,))
        lg = T.stack([kk @ wv for kk in Ks], dim=0)          # [N+1, B*S]
        alpha = np.asarray(T.functional.softmax(lg, axis=0).to_numpy(), np.float64)
        acc = np.zeros((B * S, D), np.float64)
        for n in range(alpha.shape[0]):
            acc += alpha[n][:, None] * np.asarray(V[n].to_numpy(), np.float64)
        return T.Tensor(acc.astype(np.float32)).reshape((B, S, D))

    def _attn(self, x, i):
        cfg = self.cfg
        D, NH, NKV = cfg["d"], cfg["nh"], cfg["n_kv"]
        B, S = int(x.shape[0]), int(x.shape[1])
        DK = D // NH
        p = "blocks.%d.attn." % i
        xf = x.reshape(B * S, D)
        q = self._lin(xf, p + "q_proj.weight").reshape(B, S, NH, DK).permute(0, 2, 1, 3).reshape(B * NH, S, DK)
        k = self._lin(xf, p + "k_proj.weight").reshape(B, S, NKV, DK).permute(0, 2, 1, 3).reshape(B * NKV, S, DK)
        v = self._lin(xf, p + "v_proj.weight").reshape(B, S, NKV, DK).permute(0, 2, 1, 3).reshape(B * NKV, S, DK)

        q = self._rms(q, p + "q_norm.weight")        # QK-Norm 在 RoPE 之前
        k = self._rms(k, p + "k_norm.weight")

        # 官方 buffer 形状 [1,1,max_seq,dh]，取前 S 个位置并平铺到 [B*NH, S, dh]
        def _cs(name, nh):
            a = np.asarray(self.w[p + name].to_numpy(), np.float32).reshape(-1, DK)[:S]
            return T.Tensor(np.tile(a, (B * nh, 1, 1)))

        q = q * _cs("rope.cos", NH) + _rh(q) * _cs("rope.sin", NH)
        k = k * _cs("rope.cos", NKV) + _rh(k) * _cs("rope.sin", NKV)

        if NKV != NH:
            rep = NH // NKV
            def _rep(t):
                parts = []
                for h in range(NKV):
                    one = t[h:h + 1]
                    for _ in range(rep):
                        parts.append(one)
                return T.cat(parts, dim=0)
            k = _rep(k)
            v = _rep(v)

        o = T.functional.scaled_dot_product_attention(q, k, v, causal=True)
        o = o.reshape(B, NH, S, DK).permute(0, 2, 1, 3).reshape(B * S, D)
        return self._lin(o, p + "o_proj.weight").reshape(B, S, D)

    def _moe(self, x, i):
        cfg = self.cfg
        D, E, K = cfg["d"], cfg["E"], cfg["topk"]
        B, S = int(x.shape[0]), int(x.shape[1])
        N = B * S
        p = "blocks.%d.ffn." % i
        xf = x.reshape(N, D)

        s = _sigmoid(self._lin(xf, p + "router.weight"))
        s_np = np.asarray(s.to_numpy(), np.float64)
        bias = np.asarray(self.w[p + "expert_bias"].to_numpy(), np.float64).reshape(-1)
        idx = np.argsort(-(s_np + bias), axis=1, kind="stable")[:, :K]
        gsel = np.take_along_axis(s_np, idx, axis=1)
        gate = gsel / np.maximum(gsel.sum(axis=1, keepdims=True), 1e-9)

        xn = np.asarray(xf.to_numpy(), np.float64)
        out = np.zeros((N, D), np.float64)
        for e in range(E):
            # 该专家在哪些 (token, slot) 上被选中
            hits = []
            for slot in range(K):
                rows = np.where(idx[:, slot] == e)[0]
                for r in rows:
                    hits.append((int(r), float(gate[r, slot])))
            if not hits:
                continue
            rows = np.array([h[0] for h in hits])
            gw = np.array([h[1] for h in hits], np.float32).reshape((-1, 1))
            sub = T.Tensor(xn[rows].astype(np.float32))
            ep = p + "experts.%d." % e
            y = self._lin(situ_glu(self._lin(sub, ep + "w1.weight"),
                                   self._lin(sub, ep + "w2.weight")), ep + "w3.weight")
            yn = np.asarray(y.to_numpy(), np.float64) * gw.astype(np.float64)
            out[rows] += yn
        out_t = T.Tensor(out.astype(np.float32))

        out_t = self._rms(out_t, p + "post_norm.weight")     # 聚合后归一化
        sp = p + "shared_experts.0."
        sh = self._lin(situ_glu(self._lin(xf, sp + "w1.weight"),
                                self._lin(xf, sp + "w2.weight")), sp + "w3.weight")
        return (out_t + sh).reshape(B, S, D)                 # 共享专家无门控相加

    def forward(self, ids):
        cfg = self.cfg
        D, L, rb = cfg["d"], cfg["L"], cfg["res_block"]
        ids = np.asarray(ids, dtype=np.int64)
        B, S = int(ids.shape[0]), int(ids.shape[1])

        emb = T.nn.Embedding(cfg["V"], D)
        emb.weight = self.w["embed.weight"]
        partial = emb(ids).reshape(B, S, D)

        blocks = []
        for i in range(L):
            n = self._attn_res(blocks, partial, "attn_res_proj", i)
            a = self._attn(self._rms(n, "blocks.%d.ln1.weight" % i), i)
            new_partial = partial + a
            if (i + 1) % rb == 0:
                blocks.append(new_partial)
            partial = new_partial

            n = self._attn_res(blocks, partial, "mlp_res_proj", i)
            m = self._moe(self._rms(n, "blocks.%d.ln2.weight" % i), i)
            partial = partial + m

        x = self._rms(partial, "final_norm.weight")
        logits = x.reshape(B * S, D) @ self.w["lm_head.weight"].T
        return logits.reshape((B, S, int(self.w["lm_head.weight"].shape[0])))


def load_oracle(prefix):
    W = np.load(prefix + "weights.npz")
    D = int(W["embed.weight"].shape[1])
    nh = 4
    k = W["blocks.0.attn.k_proj.weight"].shape[0]
    cfg = dict(V=int(W["embed.weight"].shape[0]), d=D, nh=nh, n_kv=int(k) // (D // nh),
               L=len(set(int(s.split(".")[1]) for s in W.files if s.startswith("blocks."))),
               E=W["blocks.0.ffn.router.weight"].shape[0], topk=2, res_block=2)
    return cfg, {s: W[s] for s in W.files}
