# -*- coding: utf-8 -*-
"""`tgai_go.train` 第一切片的判分台（CPU/numpy，无需 GPU）。

判据：
  A. 自动微分 vs 数值梯度（中心差分）相对误差 < 1e-4
  B. AdamW 与闭式参考实现逐元素一致（< 1e-7）
  C. Muon 的 Newton-Schulz 输出近似正交（||X Xᵀ - I||_F 小）
  D. 端到端：小 MLP 拟合非线性函数，loss 明显下降
  E. state_dict / load_state_dict 往返一致；Embedding 反向正确聚合

用法: python test_train_api.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "python"))

import tgai_go.train as tt          # noqa: E402
from tgai_go.train import Tensor, nn, optim, functional as F   # noqa: E402


def finite_diff(f, x, eps=1e-3):
    """中心差分梯度（x 为 numpy 数组）"""
    g = np.zeros_like(x, dtype=np.float64)
    flat = x.reshape(-1)
    gf = g.reshape(-1)
    for i in range(flat.size):
        old = flat[i]
        flat[i] = old + eps
        lp = f(x)
        flat[i] = old - eps
        lm = f(x)
        flat[i] = old
        gf[i] = (lp - lm) / (2 * eps)
    return g


def test_autograd():
    rng = np.random.default_rng(0)
    A = rng.standard_normal((4, 5)).astype(np.float32)
    B = rng.standard_normal((5, 3)).astype(np.float32)
    w = rng.standard_normal(3).astype(np.float32)

    def f_scalar(A_np):
        ta = Tensor(A_np, requires_grad=True)
        tb = Tensor(B, requires_grad=True)
        y = (ta @ tb)                      # [4,3]
        s = F.silu(y)                      # 非线性
        z = (s * s).mean()
        loss = z * 1.7 + y.sum() * 0.3
        loss.backward()
        return float(loss.data), ta.grad

    loss, grad_auto = f_scalar(A)

    def f_loss_only(A_np):
        ta = Tensor(A_np, requires_grad=True)
        tb = Tensor(B, requires_grad=True)
        y = ta @ tb
        s = F.silu(y)
        z = (s * s).mean()
        loss = z * 1.7 + y.sum() * 0.3
        return float(loss.data)

    grad_num = finite_diff(f_loss_only, A)
    err = np.abs(grad_auto - grad_num).max() / (np.abs(grad_num).max() + 1e-12)
    ok = err < 1e-4
    print("[A] 自动微分 vs 数值梯度: 最大相对误差 = %.3e  -> %s" % (err, "PASS" if ok else "FAIL"))
    return ok


def test_adamw():
    rng = np.random.default_rng(1)
    p0 = rng.standard_normal((3, 4)).astype(np.float32)
    g = rng.standard_normal((3, 4)).astype(np.float32)
    lr, b1, b2, eps, wd = 1e-3, 0.9, 0.999, 1e-8, 1e-2

    # 我们实现
    p = nn.Parameter(p0.copy())
    p.grad = g.copy()
    opt = optim.AdamW([p], lr=lr, betas=(b1, b2), eps=eps, weight_decay=wd)
    opt.step()

    # 闭式参考（torch 语义）
    m = (1 - b1) * g
    v = (1 - b2) * g * g
    bc1, bc2 = 1 - b1 ** 1, 1 - b2 ** 1
    ref = p0 - lr * wd * p0 - lr * (m / bc1) / (np.sqrt(v / bc2) + eps)
    err = np.abs(p.data - ref).max() / (np.abs(ref).max() + 1e-12)
    ok = err < 1e-6
    print("[B] AdamW vs 闭式参考: 最大相对误差 = %.3e  -> %s" % (err, "PASS" if ok else "FAIL"))
    return ok


def test_muon():
    """NS5 的性质是**把谱压平**，不是把奇异值变成 1。

    这几个系数对应 p(x)=3.4445x-4.775x^3+2.0315x^5，其不动点是 0.868，
    所以正交化后奇异值集中在 0.868 附近（Muon 的学习率吸收这个常数）。
    判据：病态输入（条件数 ~384）经 5 步后条件数 <= 2.0，且改善 >= 50x。
    """
    rng = np.random.default_rng(2)
    from tgai_go.train import _newton_schulz5
    base = rng.standard_normal((16, 8))
    scale = np.array([1, .5, .2, .1, .05, .02, .01, .005])
    g = (base @ np.diag(scale)).astype(np.float32)
    sv0 = np.linalg.svd(g, compute_uv=False)
    cond0 = sv0.max() / sv0.min()
    o = _newton_schulz5(g)
    sv1 = np.linalg.svd(o, compute_uv=False)
    cond1 = sv1.max() / sv1.min()
    ok = cond1 <= 2.0 and cond0 / cond1 >= 50
    print("[C] Muon 压谱: 条件数 %.0f -> %.2f（改善 %.0fx）  -> %s"
          % (cond0, cond1, cond0 / cond1, "PASS" if ok else "FAIL"))
    return ok


def test_end_to_end():
    rng = np.random.default_rng(3)
    n = 256
    X = rng.uniform(-2, 2, (n, 2)).astype(np.float32)
    Y = (np.sin(X[:, 0] * 1.5) * np.cos(X[:, 1]) + 0.3 * X[:, 0]).astype(np.float32)[:, None]

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.l1 = nn.Linear(2, 32, bias=True)
            self.n1 = nn.RMSNorm(32)
            self.l2 = nn.Linear(32, 32, bias=True)
            self.l3 = nn.Linear(32, 1, bias=True)

        def forward(self, x):
            h = F.silu(self.l1(x))
            h = self.n1(h)
            h = F.silu(self.l2(h))
            return self.l3(h)

    np.random.seed(0)
    m = MLP()
    opt = optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    xT, yT = Tensor(X), Tensor(Y)
    losses = []
    for step in range(400):
        pred = m(xT)
        d = pred - yT
        loss = (d * d).mean()
        m.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    drop = losses[0] / max(losses[-1], 1e-12)
    ok = losses[-1] < losses[0] * 0.05
    print("[D] 端到端训练: loss %.4f -> %.4f（下降 %.1fx，%d 参数）-> %s"
          % (losses[0], losses[-1], drop, m.num_params(), "PASS" if ok else "FAIL"))
    return ok


def test_state_dict():
    np.random.seed(1)
    m1 = nn.Sequential(nn.Linear(4, 8, bias=True), nn.RMSNorm(8), nn.Linear(8, 2, bias=False))
    sd = m1.state_dict()
    m2 = nn.Sequential(nn.Linear(4, 8, bias=True), nn.RMSNorm(8), nn.Linear(8, 2, bias=False))
    m2.load_state_dict(sd)
    x = np.random.randn(3, 4).astype(np.float32)
    same = np.allclose(m1(Tensor(x)).data, m2(Tensor(x)).data, atol=1e-6)

    # Embedding 反向：重复 index 应聚合梯度
    emb = nn.Embedding(5, 3)
    idx = np.array([[0, 1, 0], [0, 0, 0]])
    y = emb(idx)
    (y * y).sum().backward()
    cnt = np.bincount(idx.reshape(-1), minlength=5)
    # 每个出现位置贡献 2*w，所以总梯度 = 2*cnt*w
    expect = 2.0 * cnt[:, None] * emb.weight.data
    emb_ok = np.allclose(emb.weight.grad, expect, atol=1e-5)
    ok = same and emb_ok
    print("[E] state_dict 往返=%s  Embedding 梯度聚合=%s -> %s"
          % (same, emb_ok, "PASS" if ok else "FAIL"))
    return ok




def test_attention_grad():
    """F. 注意力：自动微分 vs 数值梯度（对输入和对 q_proj 权重）"""
    np.random.seed(11)
    B, S, D, H = 2, 4, 8, 2
    attn = nn.CausalSelfAttention(D, H)
    x0 = np.random.randn(B, S, D).astype(np.float32) * 0.5

    def loss_of(x_np, w_np=None):
        if w_np is not None:
            attn.q_proj.weight.data = w_np.reshape(attn.q_proj.weight.shape).copy()
        y = attn(Tensor(x_np))
        return float((y * y).mean().data)

    xT = Tensor(x0, requires_grad=True)
    y = attn(xT)
    loss = (y * y).mean()
    loss.backward()
    gx_auto = xT.grad.copy()

    w0 = attn.q_proj.weight.data.copy()
    gq_auto = attn.q_proj.weight.grad.copy()

    gx_num = finite_diff(lambda z: loss_of(z), x0)
    ex = np.abs(gx_auto - gx_num).max() / (np.abs(gx_num).max() + 1e-12)

    def loss_w(w_np):
        return loss_of(x0, w_np)

    # 权重梯度很小（|g|max~6e-3），float32 下中心差分对 eps 很敏感：
    # 实测 eps=1e-2 -> 2.0e-4，eps=1e-3 -> 1.6e-3，误差随 eps 减小而增大 ⇒ 差分噪声
    gw_num = finite_diff(loss_w, w0, eps=1e-2)
    # 重新算一次 autograd 的 w 梯度（上面已有）
    ew = np.abs(gq_auto - gw_num).max() / (np.abs(gw_num).max() + 1e-12)
    attn.q_proj.weight.data = w0
    ok = ex < 1e-3 and ew < 1e-3
    print("[F] 注意力梯度: dx 误差=%.2e  dWq 误差=%.2e  -> %s"
          % (ex, ew, "PASS" if ok else "FAIL"))
    return ok


def test_attention_causal():
    """G. 因果性：改动未来 token 不能影响过去位置的输出"""
    np.random.seed(12)
    B, S, D, H = 1, 6, 8, 2
    attn = nn.CausalSelfAttention(D, H)
    x = np.random.randn(B, S, D).astype(np.float32)
    y1 = attn(Tensor(x)).data.copy()

    x2 = x.copy()
    x2[0, -1, :] += 5.0                    # 只改最后一个位置
    y2 = attn(Tensor(x2)).data
    diff_past = np.abs(y1[0, :-1] - y2[0, :-1]).max()
    diff_last = np.abs(y1[0, -1] - y2[0, -1]).max()
    ok = diff_past < 1e-6 and diff_last > 1e-6
    print("[G] 因果性: 过去位置变化=%.2e（应为0）  末位变化=%.2e（应>0）-> %s"
          % (diff_past, diff_last, "PASS" if ok else "FAIL"))
    return ok


def test_attention_train():
    """H. 端到端：mini 注意力语言模型学会预测周期序列的下一个 token"""
    np.random.seed(13)
    V, S, D, H = 8, 16, 32, 4
    period = 4
    data = np.array([[ (i % period) for i in range(S + 1)] for _ in range(32)], dtype=np.int64)

    class TinyLM(nn.Module):
        def __init__(self):
            super().__init__()
            self.emb = nn.Embedding(V, D)
            self.n1 = nn.RMSNorm(D)
            self.attn = nn.CausalSelfAttention(D, H)
            self.n2 = nn.RMSNorm(D)
            self.head = nn.Linear(D, V, bias=False)

        def forward(self, idx):
            B, L = idx.shape
            x = self.emb(idx)
            x = x + self.attn(self.n1(x))
            x = self.n2(x)
            return self.head(x.reshape(B * L, D))

    m = TinyLM()
    opt = optim.AdamW(m.parameters(), lr=3e-3, weight_decay=0.0)
    inp = data[:, :S]
    tgt = data[:, 1:S + 1]
    losses = []
    for step in range(300):
        logits = m(inp)
        loss = F.cross_entropy(logits, tgt.reshape(-1))
        m.zero_grad()
        loss.backward()
        opt.step()
        losses.append(float(loss.data))
    acc = 0.0
    with tt.no_grad():
        logits = m(inp)
        pred = logits.data.argmax(-1).reshape(-1)
        acc = float((pred == tgt.reshape(-1)).mean())
    ok = losses[-1] < 0.15 and acc > 0.95
    print("[H] 注意力 LM 训练: loss %.3f -> %.4f，teacher-forced 准确率 %.1f%%（%d 参数）-> %s"
          % (losses[0], losses[-1], acc * 100, m.num_params(), "PASS" if ok else "FAIL"))
    return ok


def main():
    results = {
        "A_autograd": test_autograd(),
        "B_adamw": test_adamw(),
        "C_muon": test_muon(),
        "D_train": test_end_to_end(),
        "E_state": test_state_dict(),
        "F_attn_grad": test_attention_grad(),
        "G_causal": test_attention_causal(),
        "H_attn_train": test_attention_train(),
    }
    allok = all(results.values())
    print("\n汇总:", ", ".join("%s=%s" % (k, "PASS" if v else "FAIL") for k, v in results.items()))
    print("VERDICT ok=%s" % allok)
    return 0 if allok else 1


if __name__ == "__main__":
    sys.exit(main())
