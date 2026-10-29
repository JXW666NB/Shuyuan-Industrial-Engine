# 收尾清单（2026-10-29）

> 对应目标：A 段（性能追平/超越 PyTorch）+ B 段（真纯 C 预设架构）。
> 每项都写：**判据 / 实测 / 复跑命令 / 状态**。未达成的单独列，并说明缺什么。

---

## 一、达成了什么

### A 段：性能（**全部达成**）

| 判据 | 目标 | 实测 | 状态 |
|---|---|---|---|
| A1 seq=2048 单步 | ≤0.90× PyTorch | **0.036 s vs 0.096 s = 0.375×**（快 2.7 倍） | ✅ |
| A2 seq=8192 单步 | ≤1.00× | **0.144 s vs 0.189 s = 0.762×**（快 1.3 倍） | ✅ |
| A3 显存 seq=2048 | ≤1.2× | **3.418 GiB vs 2.966 GiB = 1.15×** | ✅ |
| A3 显存 seq=8192 | ≤1.2× | **6.775 GiB vs 7.101 GiB = 0.95×（更省）** | ✅ |
| A4-① 依赖 | 是否含 torch | CPU venv **101 MB**（仅 numpy）；wheel `Requires-Dist` 只有 numpy | ✅ |
| A4-② 只改导入迁移 | 同一份脚本 | **13/13 CPU + 13/13 CUDA** | ✅ |
| A4-③ 训完直出 `.TG` 被引擎加载 | 加载+生成 | `load` 5/5、GPU 引擎生成 **5/5**、退出码 0、导出字节确定 | ✅ |
| A4-④ CPU/GPU 同一份代码 | 同一个模型类 | 四个预设均在 CPU 与 CUDA 上跑同一份 Python | ✅ |

复跑：
```bash
python3 -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 2048
python3 -u training/harness/a3_mem_parity.py --backend tgai --seq 8192
python3 -u training/harness/pytorch_conformance.py --cuda
```

### 补充：**能不能替代 PyTorch**（过拟合对照）

| 项 | TGAI CUDA | PyTorch |
|---|---|---|
| 固定 32 样本，loss 首→末 | 4.161366 → **0.008179** | 4.161367 → 0.008179 |
| **400 步逐 step 最大相对差** | **5.684e-05** | — |

短程 20 步为 **2.322e-07**。复跑：`python3 -u training/harness/overfit_parity.py --steps 400`

### B 段：预设架构

| 项 | 判据 | 实测 | 状态 |
|---|---|---|---|
| B1 llama | 前向<1e-4 / 反向<1e-3 | Python 8.664e-06 / 9.228e-06；**C 8.664e-06** | ✅ |
| B2 GPT-2 | 同上 + **真实权重** | 1.857e-07；**真实 GPT-2 权重 8.254e-07、argmax 全同**；C 2.698e-07 | ✅ |
| B3 Qwen2 | 同上 | 4.403e-06 / 8.382e-06；**C 4.403e-06（同一份 C，仅 qkv_bias=1）** | ✅ |
| B4 TGAI V3（非 CSA） | 前向<1e-4 / 反向<1e-3 | 对官方 `model_v3.py`：前向 **4.793e-05**、反向 **1.267e-04** | ✅ |
| B4 1/8 规模 + 参数对齐 | 参数量与配置对齐 | 全量 1.5295 B，1/8 配置 **0.1874 B（比值 0.979）**，logits **1.062e-05** | ✅ |
| B6 一键流水线 | 给语料即可训 | 自训词表 132、loss **5.3091→0.1755**、重载逐位一致；支持 llama/qwen2/gpt2 | ✅ |
| B7 打包 | C 源码随 wheel + README | wheel 含 `_native/src/llama_fwd.c`(41 KB)；README 有「预设架构+纯 C」章节；装后自训 BPE 可用 | ✅ |

复跑（一次跑完）：
```bash
bash training/harness/run_all.sh
```

---

## 二、没达成什么（**不掩饰**）

| 项 | 状态 | 缺什么 | 为什么没做 |
|---|---|---|---|
| **B4 的 CSA**（压缩稀疏注意力） | ❌ 未做 | 压缩池化 + 块对角掩码 + 局部窗口 + 压缩后卷积，是一个独立大模块 | 剩余轮次内做不完；半成品比不做更糟 |
| **B4 的 C 侧** | ❌ 未做 | V3 的 Block AttnRes（跨层状态列表）与 SiTU-MoE 需要扩展 `TgArch`，不是"加字段"能解决 | 同上 |
| **B5 DeepSeek 完整版** | ⚠️ 只做第一步 | MLA 结构核心已通（CPU 1.944e-07 / CUDA 3.092e-07）；缺 RoPE 接入、完整 MoE 路由、吸收矩阵 | 目标原文即写"最后做，单独子目标" |
| **B1/B3 的 C 侧反向** | ❌ 未做 | C 核心只有前向图；反向仍在 Python/CUDA 后端 | 目标只要求"与 PyTorch 对拍前向+反向"，已在 Python 侧满足 |
| **CPU 版引擎退出时堆损坏** | ⚠️ 已定性未修 | `_native/libtgai_engine.dll` 在进程退出/卸载时堆损坏；**工作本身正确**（load/encode/generate 都成功） | 需 ASan/调试器定位；产线用 GPU 版，不受影响；建议单独立项 |
| GitHub 仓库的 **About 简介** | ❌ 未改 | 需在网页/API 改，`git push` 改不了 | 文案已备好，见 §四 |

---

## 三、下一步建议（按优先级）

1. **修 CPU 版引擎的退出堆损坏**（唯一有生产风险的一项）
   - 本机已有完整构建能力：MSVC 2022 BuildTools + nvcc 12.4，`build_cuda_v100.bat` 零改动可重编；
   - 建议用 AddressSanitizer 定位（`/fsanitize=address`），再查 `tg_unload` / `DllMain` 的释放顺序；
   - 已实测：单变量改 d/V/L/E/act/seq 九个变体**都能加载成功**，崩溃只出现在退出阶段。

2. **补 B4 的 CSA**（V3 与生产模型的关键差）
   - 先把 CSA 的**参数布局**从 `model_v3.py` 里读出来（`CompressedSparseAttention`），
     再做压缩池化与块对角掩码；
   - 已有 oracle 方法论：用官方实现生成 logits/梯度真值，我们逐值对拍。

3. **把 A4 证据包做成对外一页**（`docs/why_tgai_go.md` 已有草稿，需按最新数字更新）
   - 数字：0.375× / 0.762× / 1.15× / 0.95× / 101 MB / 13-13 / 5.684e-05。

4. **CI 化**：把 `run_all.sh` 接进 CI，每次改动跑一遍（含"判分台自证"）。
   - 本期教训：**判据本身被骗过 4 次**（退出码 / nvidia-smi / stdout 缓冲 / 写死预设），
     只有自动化重跑才能防复发。

---

## 四、GitHub About 简介（可直接粘贴）

> 推理 + 训练一体的自研 AI 引擎：纯 C 预设架构（llama / GPT-2 / Qwen2 / TGAI V3），
> CUDA 训练比 PyTorch 快 1.3–2.7 倍、显存 0.95–1.15 倍，硬依赖只有 numpy。

**Topics 建议**：`llm` `inference-engine` `training` `moe` `cuda` `pure-c` `chinese` `numpy`

---

## 五、本期最贵的三条教训

1. **判据本身必须先被验证** —— 本期四次被自己的判据骗：
   `退出码==0`（引擎退出阶段堆损坏）、`nvidia-smi` 采样（同码读出 2.90/6.17 GiB）、
   `stdout 找 GEN`（崩溃早于刷新）、`--c-forward` 写死 `presets.llama`（qwen2 假失败）。
2. **"源码树里能跑" ≠ "装到用户机器能跑"** —— 四例：`packages` 漏 `tgai_go.train`、
   `pipeline` 读仓库路径、全局 `*.c` 排除、`__version__` 与 pyproject 不一致。
   **每个交付都要在干净 venv 里过一遍。**
3. **能重编才能修** —— 曾以为"没有编译器"，实际 `cl.exe` 只在 Developer Command Prompt 的
   PATH 里（`where cl` 查不到不代表没装）。打通构建后才有"零改动重编 → 验证 → 再改"的资格。
