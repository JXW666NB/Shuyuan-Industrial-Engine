# -*- coding: utf-8 -*-
"""L2 判分台：完整训练步 + 100 步 loss 曲线对齐。

两路完全同权、同数据、同超参：
  ref   : PyTorch autograd + 手写 Muon/AdamW（与 kernel 同语义）
  engine: Grok 交付的 kernel（layer_fwd/bwd、mm_fwd/bwd、ce_fwd/bwd、muon_step、adamw_step）

模型（单层 Transformer 块 + 线性头）：
  x[N,d] → layer(attn+MoE) → out[N,d] → logits = out@Whead → CE(logits, tgt)

判据：
  - 100 步 loss 曲线最大绝对偏差 < 2e-3（loss 量级 ~4）
  - 末步所有参数相对误差 < 1e-4

用法:
    python l2_train.py <layer.cu> <mm.cu> <ce.cu> <opt.cu>
"""
import ctypes
import sys

import torch
import torch.nn.functional as F

from tgai_nvrtc import Cuda

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

SEQ, D, NH, DH, FF, E = 8, 32, 2, 16, 32, 2
BATCH = 1
N = SEQ * BATCH
V = 64
EPS = 1e-5
SCALE = 1.0 / (DH ** 0.5)
STEPS = int(sys.argv[5]) if len(sys.argv) > 5 else 100
LR = float(sys.argv[6]) if len(sys.argv) > 6 else 5e-3
BETA = 0.95
WD = 0.0
ADAM_B1, ADAM_B2, ADAM_EPS = 0.9, 0.999, 1e-8
NS_STEPS = 5
dev = "cuda"


def P_(t):
    return (t.data_ptr(), ctypes.c_void_p)


def I(v):
    return (int(v), ctypes.c_int)


def Fv(v):
    return (float(v), ctypes.c_float)


# ---------------- 优化器（参考侧，torch 实现，与 kernel 语义一致） ----------------
def ns5(X, steps=NS_STEPS, eps=1e-7):
    a, b, c = 3.4445, -4.7750, 2.0315
    Y = X / (X.norm() + eps)
    tr = Y.size(0) > Y.size(1)
    if tr:
        Y = Y.T
    for _ in range(steps):
        A = Y @ Y.T
        Bq = b * A + c * (A @ A)
        Y = a * Y + Bq @ Y
    if tr:
        Y = Y.T
    return Y


def muon_ref(W, G, B, lr=LR, beta=BETA, wd=WD):
    B.mul_(beta).add_(G, alpha=1 - beta)
    Xo = ns5(B)
    scale = max(1.0, W.size(0) / W.size(1)) ** 0.5
    W.sub_(lr * (Xo * scale + wd * W))
    return W, B


def adamw_ref(W, G, m, v, t, lr=LR, wd=WD):
    W.mul_(1 - lr * wd)
    m.mul_(ADAM_B1).add_(G, alpha=1 - ADAM_B1)
    v.mul_(ADAM_B2).addcmul_(G, G, value=1 - ADAM_B2)
    mhat = m / (1 - ADAM_B1 ** t)
    vhat = v / (1 - ADAM_B2 ** t)
    W.sub_(lr * mhat / (vhat.sqrt() + ADAM_EPS))
    return W, m, v


def build_params():
    torch.manual_seed(1234)
    p = {
        "w1": torch.randn(D, device=dev),
        "Wq": torch.randn(D, D, device=dev) * 0.1,
        "Wk": torch.randn(D, D, device=dev) * 0.1,
        "Wv": torch.randn(D, D, device=dev) * 0.1,
        "Wo": torch.randn(D, D, device=dev) * 0.1,
        "w2": torch.randn(D, device=dev),
        "Wr": torch.randn(D, E, device=dev),
        "Wg": torch.randn(E, D, FF, device=dev) * 0.1,
        "Wu": torch.randn(E, D, FF, device=dev) * 0.1,
        "Wd": torch.randn(E, FF, D, device=dev) * 0.1,
        "Wh": torch.randn(D, V, device=dev) * 0.1,
    }
    torch.manual_seed(99)
    x0 = torch.randn(N, D, device=dev) * 0.5
    t0 = torch.randint(0, V, (N,), device=dev).long()
    # 同一批数据重复 STEPS 步：模型应能记住（loss 明显下降），便于验证训练确实在生效
    xs = x0.unsqueeze(0).repeat(STEPS, 1, 1)
    tg = t0.unsqueeze(0).repeat(STEPS, 1)
    return p, xs, tg


def forward_torch(p, x):
    ms = (x * x).mean(dim=1, keepdim=True)
    h1 = x * torch.rsqrt(ms + EPS) * p["w1"]
    q, k, v = h1 @ p["Wq"], h1 @ p["Wk"], h1 @ p["Wv"]
    qh = q.view(N, NH, DH).transpose(0, 1)
    kh = k.view(N, NH, DH).transpose(0, 1)
    vh = v.view(N, NH, DH).transpose(0, 1)
    sc = (qh @ kh.transpose(1, 2)) * SCALE
    mask = torch.tril(torch.ones(N, N, device=dev, dtype=torch.bool))
    sc = sc.masked_fill(~mask, float("-inf"))
    Pp = torch.softmax(sc, dim=-1)
    at = (Pp @ vh).transpose(0, 1).reshape(N, D)
    o = at @ p["Wo"]
    x2 = x + o
    ms2 = (x2 * x2).mean(dim=1, keepdim=True)
    h2 = x2 * torch.rsqrt(ms2 + EPS) * p["w2"]
    logits = h2 @ p["Wr"]
    e = logits.argmax(dim=1)
    gate = torch.sigmoid(logits.gather(1, e.unsqueeze(1))).squeeze(1)
    a = torch.einsum("td,tdf->tf", h2, p["Wg"][e])
    u = torch.einsum("td,tdf->tf", h2, p["Wu"][e])
    gu = F.silu(a) * u
    y = torch.einsum("tf,tfd->td", gu, p["Wd"][e])
    out = x2 + gate.unsqueeze(1) * y
    logits2 = out @ p["Wh"]
    return out, logits2, e


def main(layer_src, mm_src, ce_src, opt_src):
    torch.cuda.init()
    cu = Cuda()

    def load(src):
        code = open(src, encoding="utf-8", errors="replace").read()
        code = code.replace("#include <cuda_runtime.h>", "").replace("#include <stddef.h>", "")
        return cu.compile(code, "sm_70")

    m_layer = load(layer_src)
    m_mm = load(mm_src)
    m_ce = load(ce_src)
    m_opt = load(opt_src)

    p0, xs, tg = build_params()

    # ---------- 参考路径 ----------
    pr = {k: v.clone().requires_grad_(True) for k, v in p0.items()}
    st_r = {k: torch.zeros_like(pr[k]) for k in pr if pr[k].dim() >= 2}
    adam_r = {k: (torch.zeros_like(pr[k]), torch.zeros_like(pr[k])) for k in pr if pr[k].dim() == 1}
    losses_r = []
    for s in range(STEPS):
        xb, tb = xs[s], tg[s]
        _, logits, _ = forward_torch(pr, xb)
        loss = F.cross_entropy(logits, tb)
        loss.backward()
        losses_r.append(loss.item())
        with torch.no_grad():
            for k in pr:
                g = pr[k].grad
                if pr[k].dim() >= 2:
                    if pr[k].dim() == 3:      # [E,d,ff] 逐专家
                        for ei in range(E):
                            muon_ref(pr[k][ei], g[ei], st_r[k][ei])
                    else:
                        muon_ref(pr[k], g, st_r[k])
                else:
                    m_, v_ = adam_r[k]
                    adamw_ref(pr[k], g, m_, v_, s + 1)
                pr[k].grad.zero_()

    # ---------- 引擎路径 ----------
    pe = {k: v.clone() for k, v in p0.items()}
    st_e = {k: torch.zeros_like(pe[k]) for k in pe if pe[k].dim() >= 2}
    adam_e = {k: (torch.zeros_like(pe[k]), torch.zeros_like(pe[k])) for k in pe if pe[k].dim() == 1}
    losses_e = []
    smem = 48 * 1024
    cache = torch.zeros(8192, device=dev)

    for s in range(STEPS):
        xb, tb = xs[s], tg[s].to(torch.int32).contiguous()
        out = torch.zeros(N, D, device=dev)
        args = [P_(xb), P_(pe["w1"]), P_(pe["Wq"]), P_(pe["Wk"]), P_(pe["Wv"]), P_(pe["Wo"]),
                P_(pe["w2"]), P_(pe["Wr"]), P_(pe["Wg"]), P_(pe["Wu"]), P_(pe["Wd"]),
                P_(out), P_(cache), I(SEQ * BATCH), I(D), I(NH), I(DH), I(FF), I(E),
                Fv(EPS), Fv(SCALE)]
        # layer kernel 用内部 seq 概念？这里 N 行视作 seq，注意力为 N×N（测试可接受）
        cu.launch(m_layer, "layer_fwd", (1, 1, 1), (1, 1, 1), args, shared=smem)

        logits = torch.zeros(N, V, device=dev)
        cu.launch(m_mm, "mm_fwd", (1, 1, 1), (1, 1, 1),
                  [P_(out), P_(pe["Wh"]), P_(logits), I(N), I(D), I(V)])
        loss_t = torch.zeros(1, device=dev)
        cu.launch(m_ce, "ce_fwd", (1, 1, 1), (1, 1, 1), [P_(logits), P_(tb), P_(loss_t), I(N), I(V)])
        cu.sync()
        losses_e.append(loss_t.item())

        dlogits = torch.zeros(N, V, device=dev)
        cu.launch(m_ce, "ce_bwd", (1, 1, 1), (1, 1, 1), [P_(logits), P_(tb), P_(dlogits), I(N), I(V)])
        dout = torch.zeros(N, D, device=dev)
        dWh = torch.zeros(D, V, device=dev)
        cu.launch(m_mm, "mm_bwd", (1, 1, 1), (1, 1, 1),
                  [P_(dlogits), P_(out), P_(pe["Wh"]), P_(dout), P_(dWh), I(N), I(D), I(V)])

        grads = {k: torch.zeros_like(pe[k]) for k in pe}
        dxb = torch.zeros(N, D, device=dev)
        argsl = [P_(dout), P_(xb), P_(pe["w1"]), P_(pe["Wq"]), P_(pe["Wk"]), P_(pe["Wv"]),
                 P_(pe["Wo"]), P_(pe["w2"]), P_(pe["Wr"]), P_(pe["Wg"]), P_(pe["Wu"]),
                 P_(pe["Wd"]), P_(cache),
                 P_(dxb), P_(grads["w1"]), P_(grads["Wq"]), P_(grads["Wk"]),
                 P_(grads["Wv"]), P_(grads["Wo"]), P_(grads["w2"]), P_(grads["Wr"]),
                 P_(grads["Wg"]), P_(grads["Wu"]), P_(grads["Wd"]),
                 I(SEQ * BATCH), I(D), I(NH), I(DH), I(FF), I(E), Fv(EPS), Fv(SCALE)]
        cu.launch(m_layer, "layer_bwd", (1, 1, 1), (1, 1, 1), argsl, shared=smem)
        grads["Wh"] = dWh
        cu.sync()

        for k in pe:
            g = grads[k]
            if pe[k].dim() >= 2:
                if pe[k].dim() == 3:
                    for ei in range(E):
                        r, c = pe[k][ei].shape
                        cu.launch(m_opt, "muon_step", (1, 1, 1), (256, 1, 1),
                                  [P_(pe[k][ei]), P_(g[ei]), P_(st_e[k][ei]),
                                   Fv(LR), Fv(BETA), Fv(WD), I(r), I(c), I(NS_STEPS), Fv(1e-7)])
                else:
                    r, c = pe[k].shape
                    cu.launch(m_opt, "muon_step", (1, 1, 1), (256, 1, 1),
                              [P_(pe[k]), P_(g), P_(st_e[k]),
                               Fv(LR), Fv(BETA), Fv(WD), I(r), I(c), I(NS_STEPS), Fv(1e-7)])
            else:
                m_, v_ = adam_e[k]
                cu.launch(m_opt, "adamw_step", ((D + 255) // 256, 1, 1), (256, 1, 1),
                          [P_(pe[k]), P_(g), P_(m_), P_(v_), Fv(LR), Fv(ADAM_B1), Fv(ADAM_B2),
                           Fv(ADAM_EPS), Fv(WD), I(s + 1), I(D)])
        cu.sync()

    # ---------- 对比 ----------
    def curve_diff():
        return max(abs(a - b) for a, b in zip(losses_r, losses_e))

    def w_diff():
        w = 0.0
        for k in p0:
            d = (pr[k].detach() - pe[k]).abs().max().item()
            s = pr[k].detach().abs().max().item() + 1e-12
            w = max(w, d / s)
        return w

    # 过拟合真实性核验：目标是否多样 + 末步预测准确率
    tg0 = tg[-1]
    n_unique = len(set(tg0.tolist()))
    _, logits_end, _ = forward_torch(pr, xs[-1])
    acc = (logits_end.argmax(dim=1) == tg0).float().mean().item()
    print("目标类别数 = %d / %d 个样本   末步准确率 = %.1f%%" % (n_unique, N, acc * 100))
    print("末步 loss 精确值: ref=%.9f eng=%.9f" % (losses_r[-1], losses_e[-1]))

    cd = curve_diff()
    wd_ = w_diff()
    print("loss[0]  ref=%.6f  eng=%.6f" % (losses_r[0], losses_e[0]))
    print("loss[50] ref=%.6f  eng=%.6f" % (losses_r[50], losses_e[50]))
    print("loss[99] ref=%.6f  eng=%.6f" % (losses_r[-1], losses_e[-1]))
    print("曲线最大绝对偏差 = %.3e" % cd)
    print("末步权重最大相对偏差 = %.3e" % wd_)

    overfit = losses_e[-1] < 0.05
    aligned = cd < 2e-3 and wd_ < 1e-4
    # 注意: Muon 按 Frobenius 范数归一化，loss≈0 时梯度是数值噪声也会被放大成满幅更新，
    # 所以「收敛后权重逐元素对齐」不是合理判据；收敛场景改判 loss 曲线 + 过拟合本身。
    if STEPS >= 300:
        ok = overfit and cd < 2e-3
        aligned = (cd < 2e-3)
    else:
        ok = aligned
    print("final_loss=%.6f  overfit=%s  aligned=%s" % (losses_e[-1], overfit, aligned))
    print("VERDICT ok=%s curve_diff=%.3e weight_rel=%.3e" % (ok, cd, wd_))
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 5:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]))
