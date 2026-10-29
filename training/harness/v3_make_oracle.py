# -*- coding: utf-8 -*-
"""用 V3 官方实现（model_v3.py）生成"金标准"：小配置的 logits + 权重。

产出（默认写到 D:\\kernel_lab\\v3_oracle\\）：
  cfg.json      解析后的关键配置
  ids.npy       输入 token
  logits.npy    官方前向输出（float32）
  weights.npz   全部参数（float32），供我们这边等价实现加载
"""
import json
import os
import sys

import numpy as np
import torch

SRC = r"D:\TGAI-V3-1.5B\03_src"
sys.path.insert(0, SRC)

OUT = r"D:\kernel_lab\v3_oracle"
os.makedirs(OUT, exist_ok=True)

from model_v3 import TGAIV3Config, TGAIV3Model  # noqa: E402

torch.manual_seed(0)
np.random.seed(0)

cfg = TGAIV3Config()
# 小配置（1/8 规模的再缩小版，只为对拍）
cfg.vocab_size = 64
cfg.d_model = 32
cfg.n_layers = 4
cfg.n_heads = 4
cfg.n_kv_heads = 2
cfg.d_ff = 32
cfg.max_seq_len = 64
cfg.n_experts = 4
cfg.n_activated = 2
cfg.n_shared_experts = 1
cfg.attn_res_block = 2
cfg.csa_compress = 1          # 关掉 CSA（本轮只验非 CSA 部分）
cfg.dropout = 0.0
cfg.hard_max_seq = 0
cfg.pad_token_id = 0      # 小词表下必须落在范围内

flags = {}
for name in ("use_csa", "dense_ffn", "attn_res_enabled", "dense_attn_res_free",
             "moe_use_situ", "moe_post_norm", "attn_qk_norm", "tie_embeddings"):
    try:
        flags[name] = getattr(cfg, name)
    except Exception as ex:
        flags[name] = "ERR:%r" % (ex,)
flags["is_global_layer"] = [bool(cfg.is_global_layer(i)) for i in range(cfg.n_layers)]
flags["rope_cfg_global"] = list(cfg.rope_cfg(True))
flags["rope_cfg_local"] = list(cfg.rope_cfg(False))
print(json.dumps(flags, ensure_ascii=False, indent=1, default=str))

model = TGAIV3Model(cfg)
model.eval()
print("参数量 %.3f M" % (sum(p.numel() for p in model.parameters()) / 1e6))

g = np.random.RandomState(1)
ids = g.randint(0, cfg.vocab_size, size=(1, 16)).astype(np.int64)
t = torch.tensor(ids, dtype=torch.long)

with torch.no_grad():
    out = model(t)
logits = out[0] if isinstance(out, (tuple, list)) else out
logits = logits.float().numpy()
print("logits", logits.shape, "sum=%.6f" % float(np.abs(logits).sum()))

np.save(os.path.join(OUT, "ids.npy"), ids)
np.save(os.path.join(OUT, "logits.npy"), logits)
w = {k: v.detach().float().numpy() for k, v in model.state_dict().items()}
np.savez(os.path.join(OUT, "weights.npz"), **w)
with open(os.path.join(OUT, "cfg.json"), "w", encoding="utf-8") as f:
    json.dump({"flags": {k: v for k, v in flags.items()},
               "tiny": {"vocab": 64, "d": 32, "L": 4, "nh": 4, "n_kv": 2, "ff": 32,
                        "seq": 16, "E": 4, "topk": 2, "shared": 1, "res_block": 2,
                        "theta": cfg.rope_theta, "eps": cfg.norm_eps}},
              f, ensure_ascii=False, indent=1)
print("已写入", OUT)
print("权重键示例:", sorted(w.keys())[:6])
