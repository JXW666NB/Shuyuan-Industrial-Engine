# C 核心：架构描述符（llama / qwen2）

同一份 `native/llama_fwd.c`。差异不复制算子，只读 `TgArch`。

## 描述符

`TgArch` 嵌在 `TgaiLlamaPlan` 里，一次 `forward` 仍然只进 C 一次。枚举从 1 起，全 0 直接拒绝，不会被当成 llama。

| 字段 | 本轮取值 | 作用 |
|---|---|---|
| `qkv_bias` | 0 / 1 | 0 = llama，投影后不加 bias；1 = qwen2，Q/K/V 投影后加 bias |
| `norm` | `TG_NORM_RMS` (1) | 只实现 RMSNorm(eps=1e-6) |
| `pos` | `TG_POS_ROPE` (1) | 只实现 rotate-half RoPE |
| `ffn` | `TG_FFN_SWIGLU` (1) | 只实现 SwiGLU |
| `n_kv` | 整数 | GQA。repeat = `nh / n_kv` |

Qwen2 的 bias 加在 `x @ W.T` 之后、拆头和 RoPE 之前，和 `llama_ref.py` 里 `qf = qf + q.bias` 的顺序一致。用的是已经在用的 `bcast_add4`：`[B*S, N] + [N]`，步长和 `CudaRuntime.add_broadcast` 相同。Q 的 N 是 `d`，K/V 的 N 是 `n_kv * dh`。原地加，不另占 scratch。

权重表前 6 项不变（tok_emb、final_norm、lm_head、cos、sin、R）。每层仍是 9 个矩阵。`qkv_bias=1` 时这一层再接 `q.bias`、`k.bias`、`v.bias`。llama 的条数和顺序与改前相同。

## Python

```python
from tgai_go import presets
presets.llama(cfg)    # 强制 qkv_bias=0，不看 cfg 里有没有这个键
presets.qwen2(cfg)    # 同一份 C，qkv_bias=1
```

`ctypes` 读嵌套结构是副本，描述符整份写回 `plan.arch`。加载时核对 `sizeof(TgaiLlamaPlan)`。本轮 C 侧是 232 字节（22 个 uint64 + 7 个 int32 + `TgArch` 20 字节 + 权重表指针），对不上就拒绝。

## 编译

没改编译参数，仍不链 libcuda / libcublas。

```bash
bash native/build_llama_fwd.sh
```

```bash
gcc -shared -fPIC -O2 -std=c11 -Wall -Wextra -Wno-unused-function \
    -o libtgai_llama_fwd.so llama_fwd.c -lm
```

远端已编过：`/home/featurize/tgai/native/libtgai_llama_fwd.so`。

## 判分台

`c_arch_parity.py --selftest`（C 连跑两次）：

- llama 两次差 = 0
- qwen2 两次差 = 0
- `VERDICT ok=True`

`c_arch_parity.py --compare`（C 前向 vs Python 参考，判据 `< 1e-4`）：

- llama 相对误差 = 8.587e-06 PASS
- qwen2 相对误差 = 4.226e-06 PASS
- `VERDICT ok=True`

`llama_parity.py --c-forward`（C llama vs PyTorch）logits = 8.664e-06，`ok=True`。这和改前同一条 PyTorch 对照，不是对 Python 参考的 8.587e-06。

## 回归

热路径没改。`python/tgai_go/train/cuda.py` 的 md5 是 `7ebe7be414a21f3d8b556370184e5931`，`train/__init__.py` 是 `03564865f5173e5e733d32c89e885c88`。本轮没有上传这两个文件。

| 检查 | 结果 |
|---|---|
| `pytorch_conformance` CPU | 13/13，`ok=True` |
| `pytorch_conformance --cuda` | 13/13，`ok=True` |
| `train_backend_parity --selftest` | loss / W / dW 都是 0 |
| `train_op_parity --selftest` | 10 项都是 0 |
| `train_layer_parity --selftest` | 17 项 worst = 0 |
| `train_opt_parity --selftest` | AdamW / Muon 的 loss、lin1、lin2 都是 0 |
| `llama_parity --selftest` | worst = 0 |
| `llama_parity --cuda` | logits 8.664e-06，梯度最差 `g:blocks.1.attn.q.weight` 9.228e-06，`ok=True` |
| `llama_parity --arch qwen2 --cuda` | logits 4.403e-06，梯度最差 `g:final_norm.weight` 8.338e-06，`ok=True` |
| `moe_selfcheck` | `ok=True`，`worst_cuda=1.716e-07` |
| `opt_step_parity --cuda` | AdamW / Muon 相对差 0 |
| `attn_lowmem_check` | 五条 PASS。dh=64 最差 6.503e-07；非 causal 与 dh=32 逐位 0 |

性能（`v27_moe_bench.py --backend tgai --layers 1 --steps 3`）：

- seq=2048：**0.035 s/step**。loss `10.5066 -> 6.7539 -> 5.4846`。
- seq=8192：两次 **0.142 s/step**、**0.143 s/step**。loss 都是 `10.5505 -> 7.3241 -> 5.6400`，和已验收记录的四位小数相同。招牌是 0.141 s。这三步的墙钟不走 `llama_fwd.c`，差大约 2 ms，本轮没有改训练源码。

显存（`a3_mem_parity.py --backend tgai`，分配器高水位）：

- seq=2048：**3.386 GiB**
- seq=8192：**6.650 GiB**

## 源码

- `native/llama_fwd.h`
- `native/llama_fwd.c`
- `native/build_llama_fwd.sh`
- `python/tgai_go/presets.py`
