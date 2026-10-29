# B4 反向：V3（非 CSA）梯度对齐

判分机：`/home/featurize/tgai`，CPU。本轮没有启动任何 CUDA 任务。

改动只在 `training/harness/v3_ref.py`：原 `V3Forward` 未改公式，新增 `V3Auto`。
备份：`training/harness/v3_ref.py.bak-20261029-1638`。

## 写法

`V3Auto` 与 `V3Forward` 前向语义一致，计算全部留在 `tgai_go.train` 的张量上，`loss.backward()` 能回到参数。numpy 只用来做 top-k 的下标（阶梯函数，本来没有梯度）。

| 位置 | 原 `V3Forward` | `V3Auto` |
|---|---|---|
| MoE gather/scatter | `to_numpy` 按下标取行，再写回 numpy | **掩码法**：每个专家对全部 token 算 SiTU-GLU，再用 0/1 掩码乘上该专家的门控后相加 |
| 门控 | `gate = s_sel / Σ s_sel`，算完就脱离图 | 同一公式，但 `s_sel` 用张量高级索引从 `sigmoid(router(x))` 取出，分母 `maximum(1e-9)` |
| `expert_bias` | 只加进 argsort | 仍然只加进 argsort。它是官方 buffer，不进 112 个参数梯度 |
| AttnRes | softmax 之后用 numpy 加权求和 | `stack` + `einsum('d,nbtd->nbt')` + `softmax(axis=0)` + `einsum('nbt,nbtd->btd')`。`blocks` 为空时直接返回 `partial`，所以第 0 层 attn/mlp 与第 1 层 attn 的 AttnRes 权重没有梯度，与官方一致 |
| RoPE `rotate_half` | numpy 交错赋值，梯度断开 | 偶数/奇数切片后 `stack((-odd, even), dim=-1)` 再 reshape，公式仍是 `(x1,x2,x3,x4) → (-x2,x1,-x4,x3)` |
| QK-Norm / RMSNorm / SiTU-GLU / SDPA | 已是张量算子 | 沿用同一套：`RMSNorm`、`situ_glu`、`scaled_dot_product_attention(causal=True)` |
| GQA | 手工 cat | `repeat_interleave(n_q/n_kv, dim=1)`，作用在 head 维 |
| 词表绑定 | 两份独立 Parameter | `lm_head.weight` 与 `embed.weight` 是同一个 Parameter，梯度加在一起 |
| padding | 无 | 官方 `Embedding(padding_idx=0)`：第 0 行的值照常用，但嵌入查表不给这一行回传梯度；词表头对第 0 行的梯度保留。实现是 `where(id==0, gathered.detach(), gathered)` |

小配置 N=16、E=4，掩码法把每个专家多算了未选中的 token，输出用 0 乘掉，和「只算选中 token 再 scatter」在精确算术下相同。

## 远端实测（判分命令）

```
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_grad_parity.py --selftest
PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_grad_parity.py --compare
```

- 自证：同侧两次 `worst=0.000e+00`（121 个张量，含下面多出来的 9 个），`ok=True`
- loss = **4.653217**（官方 4.653289）。绝对差 7.2e-05，相对约 1.5e-05。这是前向 logits 已有的约 5e-05 相对误差乘上游权重之后的残差，不是缺梯度。判分台的通过条件只看梯度相对误差
- 官方 112 个梯度全部给出，缺 0。另外多 9 个：8 个 `rope.cos/sin`（官方是 buffer，不在 `named_parameters` 里）和绑在同一对象上的 `lm_head.weight`。判分只遍历官方那 112 个键
- 最差 5 个（相对误差，判据 <1e-3）：

| 参数 | 相对误差 |
|---|---|
| `blocks.2.ffn.router.weight` | 1.267e-04 |
| `attn_res_proj.2.proj.weight` | 9.947e-05 |
| `attn_res_proj.2.norm.weight` | 9.264e-05 |
| `blocks.2.attn.q_norm.weight` | 8.958e-05 |
| `blocks.0.ffn.experts.0.w2.weight` | 8.649e-05 |

- `VERDICT ok=True`

本机同一文件再跑一遍也是 `ok=True`，最差 1.221e-04，loss 4.653207。两边差在 libm，都不碰 1e-3。

## 回归（CPU）

未跑 `--cuda`：任务写明本轮 CPU 就够，不占 4090。改动文件不被 llama/gpt2/C 架构 import。

| 命令 | 结果 |
|---|---|
| `v3_parity.py --compare` | 相对误差 **4.793e-05**，`ok=True`（仍是 5e-05 量级） |
| `c_arch_parity.py --compare` | llama 8.587e-06，qwen2 4.226e-06，gpt2 2.698e-07，`ok=True` |
| `llama_parity.py --cpu` | logits 4.261e-07，梯度最差 6.755e-06，`ok=True` |
| `llama_parity.py --cpu --arch qwen2` | logits 4.877e-07，梯度最差 7.781e-06，`ok=True` |
| `llama_parity.py --c-forward` | logits 8.664e-06，`ok=True` |
| `gpt2_parity.py --cpu` | logits 1.905e-07，梯度最差 3.852e-07，`ok=True` |
| `pytorch_conformance.py` | **13/13 PASS** |
| `pipeline_smoke.py --selftest` | 同 seed 差 0，`ok=True` |
| `pipeline_smoke.py --check` | loss 5.3091 → 0.1755，重载 logits 逐位一致，`ok=True` |

`llama_parity.py --c-forward --arch qwen2` 得到 4.135e-01，未通过。这条路径固定调用 `presets.llama(cfg)`，不会换成 qwen2 的 C 入口；qwen2 的 C 对拍是 `c_arch_parity --compare`，那条是 4.226e-06。本次没有改 `presets.py` 或 `llama_parity.py`。
