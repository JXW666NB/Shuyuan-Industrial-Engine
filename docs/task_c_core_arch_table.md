# 任务：把 C 核心从"只能跑 llama"扩展成"架构描述符驱动"
## 0. 纪律
- 本地 `D:\DSH-Workspace\TGAI-对话档案`，远端用 `feat.py sh/put`；远端单卡 4090。
- 包 `/home/featurize/tgai/python/tgai_go/`；C 源 `/home/featurize/tgai/native/`；
  判分台 `/home/featurize/tgai/training/harness/`。
- **不改 SSH key / 密码 / sudo**；单次 GPU 跑 ≤10 分钟。

## 1. 现状
- `native/llama_fwd.c` 已经能用**纯 C 编排**跑 llama 前向：一次 forward 只跨 Python↔C 一次，
  与官方参考对拍 8.664e-06。
- 但它是**写死 llama** 的。Qwen2 只差一个"Q/K/V 投影带 bias"，却跑不了。

## 2. 要做的事
把 `llama_fwd.c` 改造成**架构描述符驱动**：新增一个结构体（建议命名 `TgArch`），
下面这些差异全部由**字段**决定，C 里用 `if/switch` 选择，**不复制算子**：

| 字段 | 取值 | 影响 |
|---|---|---|
| `qkv_bias` | 0/1 | q/k/v 投影后是否加 bias（llama=0，qwen2=1） |
| `norm` | rmsnorm（本轮只需这一种） | 归一化方式 |
| `pos` | rope（本轮只需这一种） | 位置编码 |
| `ffn` | swiglu（本轮只需这一种） | FFN 类型 |
| `n_kv` | 整数 | GQA 比例（repeat 次数 = n_heads/n_kv） |

**验收只要求 llama 与 qwen2 两种组合**（其余字段先留位，后面 GPT-2/V3 再填）。

Python 侧 `tgai_go/presets.py` 增加：
- `presets.llama(cfg)`（已有，保持行为不变）
- `presets.qwen2(cfg)`（新增；内部就是把 `qkv_bias=1` 传给同一份 C）

## 3. 判据（判分台已写好，`training/harness/c_arch_parity.py`）
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/c_arch_parity.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/c_arch_parity.py --compare"
```
- 自证：C 连跑两次必须**逐位一致（差 0）**
- 考核：llama 与 qwen2 的 **C 前向 vs Python 参考 相对误差 < 1e-4**

## 4. 不许回归（主控每次自己复跑）
- `llama_parity.py --cuda` / `--c-forward` / `--arch qwen2 --cuda` 仍绿
- `pytorch_conformance` CPU/CUDA 均 13/13
- 四 `--selftest` 全 0；`moe_selfcheck` ok；`opt_step_parity --cuda` 逐位 0；`attn_lowmem_check` 全 PASS
- 性能与显存不许退：seq=2048 **0.035 s**、seq=8192 **0.141 s**、峰值 **3.386 / 6.650 GiB**

## 5. 交付
- C 源码改在 `native/`，编译脚本一并更新；报告 `docs/train_c_core_arch_table.md`
- 回话最后 6-8 行：两个架构的 C 前向误差、自证是否为 0、回归是否全过
