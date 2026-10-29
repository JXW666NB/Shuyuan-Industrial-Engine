# CUDA 切片与 GQA head 重复

两处缺口：CUDA 张量的 `Tensor.__getitem__` 以前直接走 `self.data[idx]`，而 CUDA 上 `data` 是 `None`；llama 的 GQA 又要把 `[B*NKV, S, Dh]` 沿 head 维连续复制成 `[B*NH, S, Dh]`。都在设备上做，不把整块张量拉回主机。

## `__getitem__`

只做 PyTorch 里最常见的基本索引：整数（丢掉这一维）、切片（保留这一维，step 可以为负）、`...`。缺的尾部维度补成全切片。`None`、布尔和数组下标仍然拒绝——Embedding 继续走已有的 row-gather，不从这里绕。

主机只做形状运算，得到一份 gather 计划 `(out_shape, base, divs, strides)`：

```
源下标 = base + Σ (输出坐标[d] // divs[d]) * strides[d]
```

切片的 `div` 恒为 1，`strides` 已经乘上 step。整数下标折进 `base`，不出现在输出形状里。前向是 `index_gather8`：每个输出元素读一次源地址，写成新的连续缓冲。

反向是 `index_scatter_add8`。先 `cuMemsetD32Async` 把与原张量同形状的缓冲清成 0，再按同一映射 `atomicAdd` 回去。基本切片的源位置互不重叠，每个位置只加一次；没被切到的位置保持 0。`slice_basic` 的检查就是这个：`x[:, :3]` 的梯度是 2，其余是 0。

lazy `.T` 的物理缓冲还是转置前那块。切片前先按逻辑形状做一次 `transpose2d`，反向写出的梯度仍是逻辑形状，交给原来的转置反向再转回去，不会转两次。

## `repeat_interleave`

选的是模块级 `repeat_interleave(input, repeats, dim=None)`，和 PyTorch 同名。`Tensor.repeat_interleave` 只是转调它。`repeats` 只接受一个整数；`dim=None` 先展平再沿 0 维重复。`repeats == 1` 原样返回。

同一套 gather。被重复的那一维 `div = repeats`，所以输出坐标 `0,1,2,3` 在 `repeats=2` 时落到源坐标 `0,0,1,1`。反向还是先清零再 `atomicAdd`：连续的 `repeats` 份梯度加回同一个源元素。CPU 用 `numpy.repeat`，反向把这一维折成 `(n, repeats)` 再求和，和 GPU 的累加是同一件事。

`llama_ref._repeat_kv` 不再用 `x[i:i+1]` 拼 `cat`（CUDA 上 `cat` 仍未实现）。`n_rep > 1` 时调用 `repeat_interleave(x, n_rep, dim=0)`。`llama_parity.CFG` 改回 `n_kv=2`（`nh=4`）。RoPE 仍用原来的 0/±1 矩阵乘，没有改成切片，避免换一条数值路径。

`slice_basic` 原来对 `randn` 出来的叶子调用 `backward()`，叶子默认 `requires_grad=False`，反向在写梯度之前就抛掉了。用例改成 `torch.randn(4, 6, requires_grad=True)`，断言本身没动。`repeat_interleave` 这条官方用例在符号存在时直接返回，计分脚本仍打印 PASS；数值由下面的 GQA 对拍和设备上的切片/重复烟雾覆盖。

## 实测

命令都在 4090 上跑，`PYTHONPATH=/home/featurize/tgai/python`。

`pytorch_conformance.py`（CPU）和 `--cuda` 都是 13/13，`VERDICT ok=True`。两条新用例都是 PASS。CUDA 片段：

```
[CUDA 后端] 已切到 device=cuda
  slice_basic        PASS
  repeat_interleave  PASS
迁移一致性: 13/13 PASS
VERDICT ok=True
```

`llama_parity.py --selftest`：CPU vs CPU `worst=0.000e+00`，`ok=True`。

`llama_parity.py --cuda`，`n_kv=2`，`nh=4`：

```
[B1 考核] llama 预设 vs PyTorch（同权重/同输入）
  logits rel = 8.664e-06   loss rel = 1.796e-06
  梯度最差 = g:blocks.1.attn.q.weight (9.228e-06)
  VERDICT ok=True（判据 logits<1e-04, grad<1e-03）
```

判据是 logits `<1e-4`、梯度 `<1e-3`。改回 GQA 之前（`n_kv=nh`）大约是 `6.7e-06 / 1.1e-05`，现在是 `8.664e-06 / 9.228e-06`，同一量级。

回归：四个 `--selftest`（backend / op / layer / opt）全是 0。`moe_selfcheck` `ok=True`，`worst_cuda=1.716e-07`。`opt_step_parity --cuda` AdamW / Muon 逐位 0。`attn_lowmem_check` 三组 dh=64 全 PASS（最差 `6.503e-07`），非 causal 与 dh=32 两条回退逐位 0。

`v27_moe_bench.py --backend tgai --layers 1 --steps 3`：

| seq | s/step | loss |
|---|---|---|
| 2048 | **0.036** | 10.5066 → 6.7539 → 5.4846 |
| 8192 | **0.170** | 10.5505 → 7.3241 → 5.6400 |

8192 的 loss 与上一轮热点记录逐位相同。0.170 s/step 就是那次记下的数字（对外说的 0.171 是同一档）。新 kernel 只在切片和 `repeat_interleave` 时发射，不进 MoE 计时循环。
