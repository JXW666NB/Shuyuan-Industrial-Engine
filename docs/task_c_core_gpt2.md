# 任务：把 GPT-2 也塞进 TgArch（描述符压力测试）
## 0. 纪律
- 本地 `D:\DSH-Workspace\TGAI-对话档案`，远端用 `feat.py sh/put`；远端单卡 4090。
- 包 `/home/featurize/tgai/python/tgai_go/`；C 源 `/home/featurize/tgai/native/`；
  判分台 `/home/featurize/tgai/training/harness/`。
- **不改 SSH key / 密码 / sudo**；单次 GPU 跑 ≤10 分钟。

## 1. 现状
`native/llama_fwd.c` 已由 `TgArch` 描述符驱动，llama 与 qwen2 共用一份 C：
判分台 `c_arch_parity.py --compare` = `llama 8.587e-06 PASS / qwen2 4.226e-06 PASS / gpt2 C 侧未实现 FAIL`。

## 2. 要做的事：让 gpt2 也走同一份 C

GPT-2 与本轮已有的 llama 系在**四个维度**不同，这正检验描述符抽象够不够：

| 维度 | llama / qwen2 | GPT-2 |
|---|---|---|
| 归一化 | RMSNorm（只有 gamma） | **LayerNorm（gamma + beta，eps=1e-5）** |
| FFN | SwiGLU（silu(g)*u） | **GELU(tanh 近似) 的两层 MLP** |
| 位置 | RoPE | **学习式位置 wpe（按位置加一张表）** |
| bias | q/k/v（qwen2）或无 | **所有线性层都带 bias**（qkv/proj/fc/proj） |

**要求**：优先**加描述符字段**（`norm` / `ffn` / `pos` 各扩一个取值 + `bias_all` 标志），
C 里用 `switch/if` 选分支；**仍然不许为 GPT-2 复制一份算子树**。
若确实发现某处无法用字段表达（例如 LayerNorm 需要 beta、GELU 需要 tanh），
请把它作为"新的原语"加进共享层（一次实现、所有架构可用），并在报告里说明。

Python 侧加 `presets.gpt2(cfg)`，走同一份 C。

## 3. 判据（判分台已扩好）
```
python feat.py sh "cd /home/featurize/tgai && bash native/build_llama_fwd.sh"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/c_arch_parity.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/c_arch_parity.py --compare"
```
- 自证：三个架构连跑两次都必须**逐位一致（差 0）**
- 考核：**llama / qwen2 / gpt2 全部 < 1e-4**

参考真值：`gpt2_ref.py` 已与 PyTorch 对拍过（CPU 1.905e-07、CUDA 1.857e-07），可直接当基准。

## 4. 不许回归（主控每次自己复跑）
- `llama_parity --cuda` / `--c-forward` / `--arch qwen2 --cuda` 仍绿
- `gpt2_parity --cuda` 仍绿（1.857e-07）
- `pytorch_conformance` CPU/CUDA 均 13/13
- 四 `--selftest` 全 0；`moe_selfcheck` ok；`opt_step_parity --cuda` 逐位 0；`attn_lowmem_check` 全 PASS
- 性能/显存不许退：seq=2048 **0.035 s**、seq=8192 **0.141 s**、峰值 **3.386 / 6.650 GiB**

## 5. 交付
报告 `docs/train_c_core_gpt2.md`：描述符新增了哪些字段、C 里怎么分支、有没有被迫复制算子、
三个架构的误差。回话最后 6-8 行：三个架构的 C 前向误差、自证是否逐位 0、回归是否全过。
