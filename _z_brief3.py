import io
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\docs\speed_optimization_brief.md"
s = io.open(p, encoding="utf-8").read()
add = '''
## ★ 第二轮修复（2026-10-30，同一轮后续）

**`cross_entropy(ignore_index=...)` 把整块 logits 拷到 CPU 算**：

```python
# 旧（__init__.py _cross_entropy_cuda_ignore）
host = Tensor(logits.to_numpy(), requires_grad=True)   # 4992x360 = 7 MB D2H
loss_cpu = _Functional._cross_entropy_cpu(host, tgt, ignore_index)
... rt.h2d(host.grad)                                  # 又 7 MB H2D
```

**修法**：用**选择矩阵**在 GPU 上算有效行。

```python
S = 单位阵取有效行   # [nvalid, n]，缓存在 GPU 上（mask 每步相同，只建一次）
lc = S @ logits                       # 只保留有效行
loss = ce_fwd(lc, tgt_valid)          # 融合 kernel
# backward: dlogits = S.T @ ce_bwd(lc, tgt_valid)，无效行天然为 0
```

**踩过的坑（重要）**：第一版用「无效行填 safe 目标 0」的偷懒法，
**梯度是对的（9.5e-08）但 loss 偏大 72%**（8.09 vs 4.69），因为那些行的 CE 被算进了均值。
必须真的只对有效行求均值。

### 效果

| 指标 | 修前 | 修后 |
|---|---|---|
| CE + backward（4992×360，20% 有效） | 8.8 ms | **0.59 ms（15×）** |
| 整步（4090） | 66.4 ms | **50.7 ms** |
| 整步（4090 实跑 600 步） | 66.5 ms | **53.3 ms** |
| 整步（V100 实跑 600 步） | 118.7 ms | **85.6 ms** |

### 累计战果

| | 最初 | bmm 修后 | CE 修后 | 对 PyTorch |
|---|---|---|---|---|
| 4090 | 609.5 ms | 66.5 ms | **53.3 ms** | 19.3× → **1.69×** |
| V100 | 588.7 ms | 118.7 ms | **85.6 ms** | 8.8× → **1.28×** |

### 剩余项（下一步）

| 项 | 现状 | 预计 |
|---|---|---|
| 反向 elementwise 融合（RMSNorm/GLU 各 5→1 kernel） | 1014 kernel/步 | 1.3–1.5× |
| `broadcast_to8` 物化（117 次/步） | 生中间张量 | 5–10% |
| 每层 fused kernel | — | 再 1.3× |
'''
if "★ 第二轮修复" not in s:
    io.open(p, "w", encoding="utf-8").write(s.rstrip() + "\n" + add)
    print("简报已加第二轮")
