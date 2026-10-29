# 任务：补 CUDA `cat`，让 MLA 的 CUDA 路径跑通
## 0. 纪律
- 本地 `D:\DSH-Workspace\TGAI-对话档案`；远端 `feat.py sh/put`。
- 包 `/home/featurize/tgai/python/tgai_go/`（尤其 `train/__init__.py`、`train/cuda.py`、`training/kernels/*.cu`）；
  判分台 `/home/featurize/tgai/training/harness/`。
- **不改 SSH key / 密码 / sudo**；单次 GPU 跑 ≤10 分钟。

## 1. 现状
`mla_parity.py --cuda` 报：

```
NotImplementedError: CUDA cat 尚未实现
  at qcat = T.cat([qn, qr], dim=-1)
```

`T.cat` 在 CPU 上已实现（`train/__init__.py` 的 `cat`），CUDA 上没有。

## 2. 要做的事
实现 **CUDA 上的 `cat`**（`tgai_go.train.Tensor` 的 `cat`，以及 `T.cat` 顶层函数走同一条路）：
- 支持沿 **任意轴** 拼接 ≥2 个张量；除拼接轴外形状必须一致；不一致要给出清晰报错
- 前向：按轴分块拷贝（可以用一个 kernel，也可以多次 `cuMemcpy`/`cuMemcpy2D`；只要结果正确且不泄漏资源）
- **反向**：把上游梯度按各自切片的偏移切回去（同样按轴）
- 数值：与 CPU 的 `cat` 逐位一致

## 3. 判据
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/mla_parity.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/mla_parity.py --cpu"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/mla_parity.py --cuda"
```
- 自证必须 **0**；`--cpu` 与 `--cuda` 均需 **< 1e-4**（CPU 现在已是 1.944e-07）

另外请自己加一条 **cat 自身的 CUDA 对拍**（沿 axis=0/1/2 各拼 2~3 个张量，与 numpy 逐位一致；
反向也要比：把上游梯度切回去）。

## 4. 不许回归
`pytorch_conformance` CPU/CUDA 均 13/13；四 `--selftest` 全 0；`moe_selfcheck` ok；
`opt_step_parity --cuda` 逐位 0；`attn_lowmem_check` 全 PASS；
`llama_parity --cuda`、`gpt2_parity --cuda`、`c_arch_parity --compare`、`v3_parity`、`v3_grad_parity`、
`pipeline_smoke --check` 仍绿；性能 **0.035 / 0.141 s**、峰值 **3.386 / 6.650 GiB** 不许退。

## 5. 交付
报告 `docs/train_cuda_cat.md`：实现方式（kernel 还是拷贝）、支持到几轴、cat 自身对拍数字、
MLA 的 CUDA 误差。回话最后 6-8 行给这些数字与回归结论。
