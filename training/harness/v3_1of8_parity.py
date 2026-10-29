# -*- coding: utf-8 -*-
"""B4 1/8 规模跑通：同一进程里官方实现 vs 我们的实现，同权重同输入比 logits。

配置（1/8 预算，官方模型数出 0.1872B，恰为 save_1_5b_exp 1.5295B 的 1/8）：
  vocab=96000 d=512 L=8 heads=8 d_ff=832 n_kv=2 E=12 act=2 shared=1
  csa_compress=4 latent=128 res_block=4 theta=5e5
为了在 CPU numpy 上能跑完，序列长度取 16。
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
from model_v3 import TGAIV3Config, TGAIV3Model

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

CFG = dict(vocab_size=96000, d_model=512, n_layers=8, n_heads=8, d_ff=832,
           max_seq_len=64, n_experts=12, n_activated=2, n_shared_experts=1,
           attn_global_every=4, n_kv_heads=2, csa_compress=1, csa_latent=128,
           csa_local_window=128, attn_res_block=4, rope_theta=500000.0)

cfg = TGAIV3Config()
for k, v in CFG.items():
    setattr(cfg, k, v)
cfg.pad_token_id = 0
torch.manual_seed(0)
model = TGAIV3Model(cfg).eval()
n_par = sum(p.numel() for p in model.parameters())
print("官方参数量 %.4f B（csa_compress=1，非 CSA）" % (n_par / 1e9))

S = 16
g = np.random.RandomState(1)
ids = g.randint(0, cfg.vocab_size, size=(1, S)).astype(np.int64)
with torch.no_grad():
    out = model(torch.tensor(ids, dtype=torch.long))
ref = out[0] if isinstance(out, (tuple, list)) else out
ref = ref.float().numpy()
print("官方前向 logits", ref.shape)

# 我们的实现：直接吃同一份 state_dict
import v3_ref
W = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
# 只保留我们认识的参数（buffer 里的 rope.cos/sin 等也在里面）
c = dict(V=cfg.vocab_size, d=cfg.d_model, nh=cfg.n_heads, n_kv=cfg.n_kv_heads or cfg.n_heads,
         L=cfg.n_layers, E=cfg.n_experts, topk=cfg.n_activated, res_block=cfg.attn_res_block)
print("我们的 cfg:", c)
t0 = time.time()
m = v3_ref.V3Forward(c, W)
got = np.asarray(m.forward(ids).to_numpy(), np.float32)
dt = time.time() - t0
num = float(np.max(np.abs(got - ref)))
den = float(np.max(np.abs(ref))) + 1e-12
print("我们的前向 %.1f s，相对误差 %.3e" % (dt, num / den))
print("VERDICT", "PASS" if num / den < 1e-4 else "FAIL")
