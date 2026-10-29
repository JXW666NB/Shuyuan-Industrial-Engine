# -*- coding: utf-8 -*-
"""量三件事，判断 seq=8192 的 1.87× 里有多少是"编排开销"：
   ① 单次 kernel 启动 + ctypes 往返的成本
   ② 一步训练里实际发生多少算子调用（前向/反向）
   ③ 该步的总时间
"""
import sys
import time

import numpy as np

sys.path.insert(0, "/home/featurize/tgai/python")
import tgai_go.train as T  # noqa: E402

T.set_device("cuda")
rt = T._cuda()

# ① 空转 kernel 的启动成本：用 memset 当最轻量的代表
n = 1024
buf = rt.zeros((n,))
K = 300
t0 = time.perf_counter()
for _ in range(K):
    rt.launch("ewise_mul_scalar", (1, 1, 1), (256, 1, 1), [("ptr", None)] * 0) if False else None
# 用真实存在的轻量算子：div_scalar
t0 = time.perf_counter()
for _ in range(K):
    rt.div_scalar(buf, 2.0)
dt = (time.perf_counter() - t0) / K
print("① 单次轻量 kernel 往返（ctypes+launch）: %.4f ms" % (dt * 1000))

# ② 数一步里的算子调用
cnt = {"n": 0}
orig = rt.launch


def counting_launch(*a, **kw):
    cnt["n"] += 1
    return orig(*a, **kw)


rt.launch = counting_launch

D, NH, FF, E, TOPK, V = 1024, 16, 4096, 8, 2, 32779
F = T.functional
T.manual_seed(7)
emb = T.nn.Embedding(V, D)
n1, n2 = T.nn.RMSNorm(D), T.nn.RMSNorm(D)
q, k, v, o = (T.nn.Linear(D, D) for _ in range(4))
moe = T.nn.MoE(D, FF, E, top_k=TOPK, gate="softmax")
nf = T.nn.RMSNorm(D)
params = [emb.weight, n1.weight, n2.weight, q.weight, k.weight, v.weight, o.weight, nf.weight]
for _n, p in moe.named_parameters():
    params.append(p)
opt = T.optim.AdamW(params, lr=3e-4)

for SEQ in (2048, 8192):
    g = np.random.RandomState(0)
    x = g.randint(0, 256, size=(1, SEQ)).astype(np.int64)
    y = g.randint(0, 256, size=(1, SEQ)).astype(np.int64)
    B, S, DH = 1, SEQ, D // NH

    # 预热一步
    for _ in range(2):
        h = emb(x)
        nn_ = n1(h)
        qq = q(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        kk = k(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        vv = v(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
        oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
        oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
        h = h + o(oo).reshape(B, S, D)
        h = h + moe(n2(h)).reshape(B, S, D)
        hn = nf(h)
        logits = hn.reshape(B * S, D) @ emb.weight.T
        loss = F.cross_entropy(logits, y.reshape(-1))
        for p in params:
            p.grad = None
        loss.backward()
        opt.step()

    cnt["n"] = 0
    t0 = time.perf_counter()
    h = emb(x)
    nn_ = n1(h)
    qq = q(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    kk = k(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    vv = v(nn_.reshape(B * S, D)).reshape(B, S, NH, DH).permute(0, 2, 1, 3).reshape(B * NH, S, DH)
    oo = F.scaled_dot_product_attention(qq, kk, vv, causal=True, lowmem=True)
    oo = oo.reshape(B, NH, S, DH).permute(0, 2, 1, 3).reshape(B * S, D)
    h = h + o(oo).reshape(B, S, D)
    h = h + moe(n2(h)).reshape(B, S, D)
    hn = nf(h)
    logits = hn.reshape(B * S, D) @ emb.weight.T
    loss = F.cross_entropy(logits, y.reshape(-1))
    n_fwd = cnt["n"]
    for p in params:
        p.grad = None
    loss.backward()
    n_all = cnt["n"]
    opt.step()
    dt_step = time.perf_counter() - t0
    est = n_all * dt
    print("② seq=%-5d 算子启动 %4d 次（前向 %d）；该步 %.3f s" % (SEQ, n_all, n_fwd, dt_step))
    print("   其中编排开销估计 %.3f s（%.0f%% 的步时间）" % (est, 100 * est / dt_step))
    print("   扣掉编排后的纯 kernel 时间估计 %.3f s" % (dt_step - est))
