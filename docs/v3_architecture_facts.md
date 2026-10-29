# TGAI V3 架构事实（读源码 `model_v3.py` 得到，不是猜的）

> 来源：`D:\TGAI-V3-1.5B\03_src\model_v3.py`（只读）。用于 B4"1/8 规模先跑通"。
> 记录时间：2026-10-29。

## 1. 配置默认值（关键项）

| 项 | 默认 | 备注 |
|---|---|---|
| `vocab_size` | 96000 | 自训中文 BPE + 字节回退 |
| `d_model / n_layers / d_ff` | 1024 / 20 / **2816** | 注意 README 写 16 层、d_ff=1664；**源码默认是 20 / 2816**，以 `save_1_5b_exp` 预设为准 |
| `n_heads / n_kv_heads` | 16 / 0 | `n_kv_heads=0` 表示等于 n_heads（MHA） |
| `attn_global_every` | 4 | 每 4 层 1 层全局注意力（`is_global_layer`） |
| `csa_compress / csa_latent` | 4 / 0 | CSA 压缩率；`csa_latent=0` → d_model//4 |
| **`attn_qk_norm`** | **True** | QK-Norm（README 未提） |
| **`tie_embeddings`** | **True** | 输出头与词表共享（README 未提） |
| `n_experts / n_activated / n_shared_experts` | 8 / 2 / **1** | 共享专家 **无门控、直接相加**（DeepSeekMoE 式） |
| **`moe_use_situ`** | **True** | 专家 FFN 用 **SiTU-GLU**，不是 SwiGLU（README 未提） |
| **`moe_post_norm`** | **True** | MoE 聚合后做 RMSNorm |
| `attn_res_block` | 4 | 每 N 层成一个 block |
| `rope_theta` | 500000 | 全局层 `rope_theta_global=5000000`（`hetero_rope` 关闭时不用） |
| `norm_eps` | 1e-6 | |

## 2. Block AttnRes（论文 arXiv:2603.15031 的 Block 版）

每层**两个** AttnRes 模块：`attn_res_proj[i]`（attn 前）与 `mlp_res_proj[i]`（mlp 前）。
每个模块 = 一个 `nn.Linear(d_model, 1, bias=False)`（学习的伪查询 w）+ 一个 RMSNorm。

对已完成的 block 表示 `blocks` 与当前 `partial`（块内累加和）做 softmax 注意力：

```
V = stack(blocks + [partial])            # [N+1, B, T, D]
K = rmsnorm_each(V)                      # 逐块归一化（与整体归一化等价，但省显存）
logits_n = w · K_n                       # 沿 D 求和 -> [N+1, B, T]
α = softmax_n(logits)                    # 在 block 维上 softmax
h = Σ_n α_n · V_n
```

`blocks` 为空时直接返回 `partial`。

## 3. 主循环调度（`TGAIV3Model.forward`）

```
x = embed(ids); blocks = []; partial = x
for i, blk in enumerate(layers):
    h = attn_res_proj[i](blocks, partial)     # ① attn 前的深度注意力
    a = blk.attn_part(h)                      # = attn(ln1(h))
    new_partial = partial + a                 # ② 块内残差累加
    if (i+1) % attn_res_block == 0:
        blocks.append(new_partial)            # ③ 到边界收进 blocks
    partial = new_partial
    h = mlp_res_proj[i](blocks, partial)      # ④ mlp 前的深度注意力
    m = blk.mlp_part(h)                       # = ffn(ln2(h))
    partial = partial + m
x = partial
```

## 4. 注意力选择

```python
if is_global(i) or not config.use_csa:
    GlobalAttention(config, is_global=is_global(i))   # 非全局层 = local-causal
else:
    CompressedSparseAttention(config)                  # CSA
```
`use_csa` 由 `csa_compress > 1` 决定。**`csa_compress=1` 即关闭 CSA** —— B4 第一阶段就走这条路。

## 5. 金标准 oracle（已生成并校验）

`training/harness/v3_make_oracle.py` 用官方实现跑小配置，导出：

| 文件 | 内容 |
|---|---|
| `training/oracle/v3_tiny_logits.npy` | 官方前向 logits `(1,16,64)`，`|logits|.sum = 96.355881` |
| `training/oracle/v3_tiny_weights.npz` | **131** 个权重张量（float32） |
| `training/oracle/v3_tiny_ids.npy` | 输入 token |
| `training/oracle/v3_tiny_cfg.json` | 关键 flags |

小配置：`vocab=64, d=32, L=4, nh=4, n_kv=2, ff=32, seq=16, E=4, topk=2, shared=1, res_block=2, csa_compress=1`
→ `use_csa=False`、`tie=True`、`situ=True`、`is_global_layer=[F,F,F,T]`，参数量 **0.077 M**。

**这是 B4 的对拍基准**：我们的实现必须复现这组 logits（判据 <1e-4）。
