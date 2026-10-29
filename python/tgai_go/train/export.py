# -*- coding: utf-8 -*-
"""把训练侧的模型导出成引擎能加载的 `.TG`。

设计要点：
1. **名字映射**：训练侧用 `nn.MoE` 的命名（`experts.{e}.gate/up/down`），
   引擎期望 `experts.{e}.w1/w2/w3` 且需要 `moe.gate.weight` 与 `moe._qb_bias`。
   映射集中在 `_ENGINE_MAP` 一处，改架构只改这张表。
2. **RoPE 的 cos/sin 由引擎按 `rope_theta` 自己算**，我们只需把 theta 传对；
   若训练侧用的是同一 theta，两边位置编码一致。
3. **词表与 merges 一起写进去**，用户不需要单独提供分词表（引擎自训 → 直接内嵌）。
4. 权重默认写 fp32（`weight_dtype=0`），便于先把数值对齐；换 fp16 是后续优化。

用法：
    from tgai_go.train.export import save_tg
    save_tg("m.TG", params, cfg, vocab_tokens, merges)
其中 params 是 {训练侧名字: numpy 数组}。
"""
import numpy as np

from ..pack import TGWriter

DT_FP32 = 0
DT_FP16 = 1


def _engine_name(name, n_experts):
    """训练侧参数名 -> 引擎期望的参数名。返回 None 表示该参数不写进 .TG。"""
    n = name
    if n in ("token_embedding.weight", "embed.weight", "wte.weight"):
        return "token_embedding.weight"
    if n in ("final_norm.weight", "ln_f.weight"):
        return "final_norm.weight"
    if n.startswith("lm_head"):
        return None                                  # 与词表共享，引擎不接受独立头
    # MoE 专家：experts.{e}.gate/up/down -> **moe.experts.{e}.w1/w2/w3**
    # ★ 引擎的命名里 `experts` 挂在 `moe.` 下面（`blocks.{i}.moe.experts.{e}.w1.weight`）。
    #   漏掉 `.moe.` 时 .TG 仍能写、能 load，但引擎按名取权重拿不到 → generate 时 access violation
    #   （实测 0x1A00）。这是 3 处命名不匹配的根因。
    if ".experts." in n:
        head, tail = n.split(".experts.", 1)
        idx, _, rest = tail.partition(".")
        rest = rest.replace(".weight", "")
        m = {"gate": "w1", "up": "w2", "down": "w3"}
        if head.endswith(".moe"):
            block = head[: -len(".moe")]
        else:
            block = head
        if rest in m:
            return "%s.moe.experts.%s.%s.weight" % (block, idx, m[rest])
        return None
    if n.endswith(".router.weight"):
        return n
    if n.endswith(".gate.weight") and ".experts." not in n:
        return n                                     # moe.gate.weight（负载均衡门控，占位）
    return n


def collect(params, n_experts, n_layers, d_model, n_heads, d_ff, vocab, max_seq,
            rope_theta=10000.0, n_activated=2, n_kv_heads=0):
    """把训练侧 dict 变成引擎可写的 (cfg, weights)。"""
    out = {}
    for k, v in params.items():
        ek = _engine_name(k, n_experts)
        if ek is None:
            continue
        out[ek] = np.ascontiguousarray(np.asarray(v, dtype=np.float32))
    # ★ 引擎会**直接按指针读** RoPE 的 cos/sin buffer，文件里缺了就是空指针：
    #   实测症状 = generate 时 `access violation reading 0x...`（加载能过，生成才崩）。
    #   这里按与 build_dummy_tg 完全相同的公式补齐。
    DH = d_model // n_heads
    half = DH // 2
    pos = np.arange(max_seq, dtype=np.float64)[:, None]
    inv = float(rope_theta) ** (-2.0 * np.arange(half, dtype=np.float64) / DH)
    ang = pos * inv[None, :]
    cos = np.concatenate([np.cos(ang)] * 2, 1).astype(np.float32).reshape(1, 1, max_seq, DH)
    sin = np.concatenate([np.sin(ang)] * 2, 1).astype(np.float32).reshape(1, 1, max_seq, DH)
    for i in range(n_layers):
        p2 = "blocks.%d.attn." % i
        if p2 + "rope.cos" not in out:
            out[p2 + "rope.cos"] = cos
        if p2 + "rope.sin" not in out:
            out[p2 + "rope.sin"] = sin

    # 引擎要求每个专家都有 `moe.gate.weight` 与 `moe._qb_bias`（负载均衡用的旁路参数）
    for i in range(n_layers):
        p = "blocks.%d.moe." % i
        if p + "gate.weight" not in out:
            if p + "router.weight" in out:
                out[p + "gate.weight"] = np.zeros_like(out[p + "router.weight"])
        if p + "_qb_bias" not in out:
            out[p + "_qb_bias"] = np.zeros(n_experts, np.float32)
    return out


def save_tg(path, params, *, n_layers, d_model, n_heads, d_ff, vocab_tokens, merges=None,
            max_seq=512, n_experts=1, n_activated=1, rope_theta=10000.0,
            special_ids=None, weight_dtype=DT_FP32, n_kv_heads=0, arch=None):
    """写 `.TG`。vocab_tokens 是词表（list[str]），merges 是 BPE 合并表。"""
    w = collect(params, n_experts, n_layers, d_model, n_heads, d_ff,
                len(vocab_tokens), max_seq, rope_theta, n_activated, n_kv_heads)
    kw = {}
    if arch is not None:
        kw["arch"] = arch
    # .TG 头部把 rope_theta 按无符号整型写（struct "<16I6Q4I"），
    # 而 θ 通常是浮点（10000.0 / 500000.0）。这里要求它是整数值再取 int，避免静默截断。
    th = float(rope_theta)
    if abs(th - round(th)) > 1e-9:
        raise ValueError("rope_theta 必须是整数值才能写进 .TG 头（收到 %r）" % (rope_theta,))
    rope_theta = int(round(th))
    wr = TGWriter(n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                  vocab=len(vocab_tokens), max_seq=max_seq,
                  n_experts=n_experts, n_activated=n_activated, rope_theta=rope_theta,
                  n_kv_heads=n_kv_heads, weight_dtype=weight_dtype, **kw)
    wr.add_weights(w)
    wr.set_tokenizer(vocab_tokens, merges=merges,
                     special_ids=special_ids or [0, 0, 0, 0])
    wr.write(path)
    return path
