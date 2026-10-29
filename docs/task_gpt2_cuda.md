# 任务：让 GPT-2 预设的 CUDA 路径跑通并与 PyTorch 对拍

## 0. 纪律
- 本地工作目录 `D:\DSH-Workspace\TGAI-对话档案`，远端用 `feat.py`：
  `python feat.py sh "命令"` / `python feat.py put <本地> <远端绝对路径>`
- 远端单卡 RTX 4090；包 `/home/featurize/tgai/python/tgai_go/`；
  判分台 `/home/featurize/tgai/training/harness/`；kernel `/home/featurize/tgai/training/kernels/`。
- **不改** SSH key / authorized_keys / 密码 / sudo；单次 GPU 跑 ≤10 分钟。
- 本地仓库 `_repos/Shuyuan-Industrial-Engine/`，改动放这里再 put 到远端验证。

## 1. 现状（我已做完的部分）
- `training/harness/gpt2_ref.py`：GPT-2 薄定义（LayerNorm / GELU-tanh / 学习式位置 / 全 bias / untied 头）
- `training/harness/gpt2_parity.py`：与 PyTorch 对拍判分台
- **CPU 已绿**：自证 `tgai CPU vs CPU = 0.000e+00`；`--cpu` 对拍 logits **2.041e-07**、梯度最差 **5.614e-07**
  （判据 logits<1e-4、梯度<1e-3）
- **CUDA 卡住**：`NotImplementedError: CUDA 后端还没有 mean`
  （`_layernorm` 用 `x.mean(-1, keepdims=True)`，CUDA 上没实现）

## 2. 要做的事

### ① CUDA `mean(axis, keepdims)` + 反向
- 前向：沿指定轴求均值（keepdim 可选）
- 反向：把上游梯度平均后广播回原形状（除以该轴长度）
- 允许只支持常见的"归约最后一维 / 全归约"，但**形状与 keepdim 语义必须对**
- 数值：与 numpy 的 `mean` 在 fp32 下同量级（<1e-6）

### ② CUDA `exp`（若缺失）
`gpt2_ref._sigmoid` 用 `(-x).exp()` 组合 sigmoid。若 CUDA 上 `exp` 未实现，请补（前向 + 反向 = 输出乘上游梯度）。
若已有，请在报告里说明。

### ③ 其它可能在 CUDA 上缺失的算子
跑 `python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/gpt2_parity.py --cuda"`
一个接一个暴露（`mean` → `exp` → 可能 `pow` / 切片 / 加法广播）。**按报错逐个补齐**，
每个都给出与 CPU 一致的数值验证。

### ④ 目标
`gpt2_parity.py --cuda` 判据全绿：**logits 相对误差 < 1e-4、梯度 < 1e-3**。

## 3. 验收（自己跑，贴数字）
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/gpt2_parity.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/gpt2_parity.py --cpu"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/gpt2_parity.py --cuda"
```

## 4. 不许回归（主控每次自己复跑）
- `pytorch_conformance` CPU 与 CUDA 均 **13/13**
- 四 `--selftest` 全 0；`moe_selfcheck` ok；`opt_step_parity --cuda` 逐位 0；`attn_lowmem_check` 全 PASS
- `llama_parity.py --cuda` 与 `--c-forward` 仍绿
- 性能招牌不许退：seq=2048 **0.035 s**、seq=8192 **0.141 s**（`v27_moe_bench.py`）
- 显存不许退：分配器峰值 2048 **3.386 GiB**、8192 **6.650 GiB**（`a3_mem_parity.py`）

## 5. 交付
- 报告 `docs/train_cuda_gpt2_ops.md`：补了哪些算子、每个的数值验证、最后三条判分台的输出
- 回话最后 6-8 行：`--cuda` 的 logits/梯度误差、补了哪几个算子、回归是否全过
