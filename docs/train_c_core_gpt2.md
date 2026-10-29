# C 核心：GPT-2 进同一份 TgArch

同一份 `native/llama_fwd.c`。llama / qwen2 / gpt2 都走这一个 `.so`。没有为 GPT-2 再复制一棵算子树。

## 描述符新增

`TgArch` 在原有 `qkv_bias / norm / pos / ffn / n_kv` 上多了一个 `bias_all`。三个枚举各加一个取值，0 仍然拒绝，不会被当成 llama。

| 字段 | 新取值 | 作用 |
|---|---|---|
| `norm` | `TG_NORM_LN` (2) | LayerNorm，gamma + beta，eps=1e-5。RMS 仍是 1，eps=1e-6 |
| `pos` | `TG_POS_WPE` (2) | 学习式位置。token embedding 之后按位置加 `wpe` 的前 S 行。RoPE 仍是 1 |
| `ffn` | `TG_FFN_GELU` (2) | 两层 MLP：`fc` → GELU(tanh 近似) → `proj`。SwiGLU 仍是 1 |
| `bias_all` | 0 / 1 | 1 时 q/k/v/o 和 FFN 的每个线性层都加 bias。qwen2 继续只用 `qkv_bias=1` |

GPT-2 的填写是：`qkv_bias=0`，`bias_all=1`，`norm=LN`，`pos=WPE`，`ffn=GELU`，`n_kv=nh`。

权重条数由这些字段算出来，对不上就拒绝：

- llama：6 + 9L（和改前一样）
- qwen2：6 + 12L（9 个矩阵后再接 q/k/v bias，和改前一样）
- gpt2：5 + 16L

头：`tok/wte`、final gamma、`lm_head`，RoPE 再接 cos/sin/R，wpe 再接一张表，LayerNorm 再接 final beta。每层按同样的规则跳过用不到的字段。GPT-2 的融合 QKV 在 Python 侧按行拆成 q/k/v 三块再上传，C 里仍然是三次 `x @ W.T`，不是另一套注意力。

`TgaiLlamaPlan` 多了四个已有 kernel 的函数指针：`ewise_exp`、`ewise_recip`、`ewise_pow`、`sum_axis8`。加载时仍核对 `sizeof`。本轮 C 侧是 272 字节（26 个 uint64 + 7 个 int32 + `TgArch` 24 字节，`w` 前补 4 字节对齐）。llama / qwen2 也把这四个指针传进去，但 RMS + RoPE + SwiGLU 的发射顺序没有多出 kernel。

## C 里怎么分支

`norm` / `pos` / `ffn` 各一个 `switch`。`bias_all` 和 `qkv_bias` 只决定要不要在投影后做已经在用的 `bcast_add4`。bias 指针是 0 就不发射，所以 llama 的 SwiGLU 内核序列和改前相同。

位置不在 Q/K 上做第二套 RoPE：`TG_POS_WPE` 时 embedding 之后加一次表，注意力里跳过 RoPE。

## 有没有被迫复制算子

没有复制算子树。LayerNorm 和 GELU 在现有 kernel 上拼不出来的那截，做成了共享原语，llama 也能调，只是本轮不用：

- LayerNorm：`sum_axis8` 求 mean/var，`ewise_pow(-0.5)` 做 rsqrt，gamma/beta 用 `broadcast_to8`。除的是 d，不是 d-1。和 `gpt2_ref._layernorm` 同一串。
- GELU(tanh)：`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`。`tanh` 用 `2*sigmoid(2z)-1`，`sigmoid` 用 `exp` 再 `pow(-1)`，因为参考是 `(exp(-z)+1)**-1`，不是除法。`ewise_recip` 接进了计划，这条路径没有用它。

没有新写 CUDA kernel，没有改 `python/tgai_go/train/cuda.py`。它的 md5 仍是 `7ebe7be414a21f3d8b556370184e5931`。

## 判分台

`c_arch_parity.py --selftest`（C 连跑两次）：

- llama 两次差 = 0
- qwen2 两次差 = 0
- gpt2 两次差 = 0
- `VERDICT ok=True`

`c_arch_parity.py --compare`（C 前向 vs Python 参考，判据 `< 1e-4`）：

- llama 相对误差 = 8.587e-06 PASS
- qwen2 相对误差 = 4.226e-06 PASS
- gpt2 相对误差 = 2.698e-07 PASS
- `VERDICT ok=True`

`llama_parity.py --c-forward`：logits 8.664e-06，`ok=True`。和上一轮同一条 PyTorch 对照。

## 回归

| 检查 | 结果 |
|---|---|
| `train_backend_parity --selftest` | loss / W / dW 都是 0 |
| `train_op_parity --selftest` | 10 项都是 0 |
| `train_layer_parity --selftest` | 17 项 worst = 0 |
| `train_opt_parity --selftest` | AdamW / Muon 的 loss、lin1、lin2 都是 0 |
| `llama_parity --selftest` | worst = 0 |
| `gpt2_parity --selftest` | worst = 0 |
| `pytorch_conformance` CPU | 13/13，`ok=True` |
| `pytorch_conformance --cuda` | 13/13，`ok=True` |
| `llama_parity --cuda` | logits 8.664e-06，梯度最差 `g:blocks.1.attn.q.weight` 9.228e-06，`ok=True` |
| `llama_parity --arch qwen2 --cuda` | logits 4.403e-06，梯度最差 `g:final_norm.weight` 8.271e-06，`ok=True` |
| `gpt2_parity --cuda` | logits 1.857e-07，梯度最差 `g:blocks.1.mlp.proj.weight` 4.321e-07，`ok=True` |
| `moe_selfcheck` | `ok=True`，`worst_cuda=1.716e-07` |
| `opt_step_parity --cuda` | AdamW / Muon 相对差 0 |
| `attn_lowmem_check` | 五条 PASS。dh=64 最差 6.503e-07；非 causal 与 dh=32 逐位 0 |

性能（`v27_moe_bench.py --backend tgai --layers 1 --steps 3`）。这条热路径不走 `llama_fwd.c`，本轮没有改它。

- seq=2048：**0.035 s/step**。loss `10.5066 -> 6.7539 -> 5.4846`。
- seq=8192：两次 **0.143 s/step**、**0.142 s/step**。loss 都是 `10.5505 -> 7.3241 -> 5.6400`。招牌 0.141 s。上一轮同一条命令也是 0.142 / 0.143，差大约 2 ms，不是这条 C 前向引进的。

显存（`a3_mem_parity.py --backend tgai`，分配器高水位）：

- seq=2048：**3.386 GiB**
- seq=8192：**6.650 GiB**

## 源码

- `native/llama_fwd.h`
- `native/llama_fwd.c`
- `native/build_llama_fwd.sh`
- `python/tgai_go/presets.py`
