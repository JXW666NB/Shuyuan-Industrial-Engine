# 运维记录：V100 GPU 无法创建新的 CUDA 上下文

> 时间：2026-10-27 21:00 左右
> 机器：DESKTOP-5F9M8MB (192.168.0.103)，Tesla V100-PCIE-32GB
> 影响：**训练无法进行**；**11450 产线未受影响**

## 症状

| 检查项 | 结果 |
|---|---|
| 产线 `11450 /v1/models` | ✅ HTTP 200，`tgai` |
| 产线推理延迟（chat 16 tokens） | ✅ **0.33 s** |
| 公网 `tgai-8i8.pages.dev/api/health` | ✅ HTTP 200 |
| `nvidia-smi` 显存 | 15.3 GB / 32 GB（16 GB 空闲） |
| GPU 利用率 | **100%**（即使产线空闲） |
| GPU 温度 | 45 ℃ |
| GPU 上进程 | 仅 `2308`（产线），`15296 MiB` |
| **新进程 `torch.randn(512,512,'cuda')`** | ❌ **卡住 >60 s 不返回** |
| **新进程 CUDA 初始化** | ❌ 卡住；其中一次直接报 `illegal memory access` |

结论：**已有上下文可用，新上下文创建挂死**。

## 诱发过程（时间线）

1. 非 tied 版 char-LM 训练两次 40,000 步，均正常完成（100%/100% 复现）
2. 改成 tied embedding 后：
   - 第一次在 step 30200 报 `illegal memory access`
   - 第二次 step 0 报同错
   - 第三次 step 2100 报同错
   - 之后**第一步 `torch.randn` 就挂住**
3. 期间我用 taskkill 清掉了自己留下的僵尸进程（928 / 2168 / 2120），但新上下文仍无法创建

## 判断

多轮 `illegal memory access` 之后，Windows 上的 NVIDIA 驱动进入了"新上下文不可用"的状态。
产线进程（2308）因上下文已建立，仍在正常服务。

## 恢复方式（需要授权，因为会短暂影响产线）

| 方案 | 影响 | 说明 |
|---|---|---|
| **A. 重启 V100 主机** | 产线中断 2–5 分钟 | 最彻底 |
| B. 停掉 11450 → `nvidia-smi --gpu-reset` → 起回 | 中断 1–2 分钟 | 需要先确认 `--gpu-reset` 在 WDDM 下可用 |
| C. 只重启产线进程 | 大概率无效 | 上下文泄漏在驱动层，不在进程层 |

> 按工作规矩：**动产线必须先取得明确授权**。所以我没有自行重启。

## 恢复后要做的事

1. 重跑 tied-embedding 训练（`python charlm_torch.py 30000 4 512 16 1024 1024 2`，带 `CUDA_LAUNCH_BLOCKING=1` 更稳）
2. `python pelican_to_tg.py pack` → `pelican_to_tg.py gen 400`，验证引擎逐字复现
3. 把结果写进 `training/README.md` 并推送

## 备注

- 训练脚本已加**每 4000 步存盘**，崩溃不再丢全部进度
- 训练崩溃与代码无关的证据：同一个脚本在 GPU 正常时跑过两次 40,000 步
