# A4-③ 结论（**第三版，最终**）：闭环在 GPU 引擎上稳定成立

记录时间：2026-10-29。本文取代同日两版较早结论。

## 1. 最终实测（本机新构建的 GPU 引擎，部署到 V100 上跑）

引擎：`build_cuda_v100.bat` **零改动**构建 → `libtgai_engine_gpu.dll`（**211,456 字节**）
部署位置：V100 `D:\kernel_lab\libtgai_engine_gpu_fresh.dll`（独立文件，未覆盖产线）

| 模型 | 生成成功 | 进程退出码 |
|---|---|---|
| 我们导出的 `.TG`（随机权重） | **5/5** | 0 |
| 我们导出的 `.TG`（全零权重） | **5/5** | 0 |

调用路径：`tgai_go.load(model, dll=<GPU DLL>)` → `Model.generate("ab", max_tokens=6)`。

**⇒ A4-③ 达成**：训练侧导出的 `.TG` 能被自家引擎加载并稳定生成。

## 2. 之前两版结论错在哪（完整纠错链）

| 版本 | 当时的结论 | 为什么错 |
|---|---|---|
| 第一版 | "子进程 0/5、全零权重段错误 ⇒ 引擎 generate 时崩" | 用**退出码==0** 判成功；引擎退出阶段堆损坏使成功也非 0 |
| 第二版 | "改成看 stdout：随机权重 5/10 ⇒ 引擎内存不安全" | 用的是**随包的 CPU 版 DLL**（`tgai_go/_native/libtgai_engine.dll`），不是产线的 GPU 引擎 |
| **第三版（本版）** | **GPU 引擎 5/5 + 退出码 0** | 用产线同款 GPU 引擎重测 |

## 3. 仍然存在的真实缺陷（不是 A4-③ 的堵点，但要修）

**CPU 版引擎**（`_native/libtgai_engine.dll`，292 KB）：
- 生成约 5/10 ~ 7/8 成功；
- 退出阶段堆损坏（`0xC0000374`）+ 运行阶段访问违例（`0xC0000005`）。

影响面：**CPU 路径**（`tgai.py`/`tgai_chat.py` 默认用它）以及任何只用 `pip install tgai-go`
而未显式指定 GPU DLL 的用户。产线 11450 用的是 GPU DLL，不受影响。

修法建议（未做）：`tg_load` 后校验 config/权重完整性（vocab>0、层数>0、每个必需权重非 NULL 且形状一致），
不通过就返回 NULL；再查退出路径的释放顺序。

## 4. 构建能力（本轮打通，重要）

| 能力 | 状态 |
|---|---|
| 本机 MSVC 2022 BuildTools | ✅ `C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\...\vcvars64.bat` |
| 本机 nvcc | ✅ CUDA 12.4 |
| `build_cuda_v100.bat` 零改动构建 | ✅ 通过（sm_70，产出 211,456 字节 DLL） |
| 在 V100 上加载并跑通 GPU forward | ✅ `[CUDA-BRIDGE] TAKEN OVER`、`QB routing bias: loaded 1/1 layers` |

**⇒ 以后改引擎可以在本机构建、传 V100 验证、再决定是否上线。**

## 5. 产线现状（未改动）

- 服务：V100 `0.0.0.0:11450`，PID 4684，用 `D:\TGAI\libtgai_engine_gpu.dll`；
- **已备份**：`D:\TGAI\libtgai_engine_gpu.dll.bak_20261029`；
- 重启方式：`schtasks /Run /TN "TGAI-GO-11450"`（技能文档 §2）；
- **本轮没有替换产线 DLL** —— 新构建只放在 `D:\kernel_lab\` 做验证。

---

## 附：CPU 版引擎崩溃的**精确刻画**（2026-10-29 补测）

之前我写过"CPU 版约 30–50% 崩溃"，措辞不准。用带 `flush` 的逐步追踪重测后，准确结论是：

| 阶段 | CPU 版（`_native/libtgai_engine.dll`） | GPU 版（产线同款） |
|---|---|---|
| `tg_load` | ✅ 成功（单变量试过 d=32/64/128/256、V=512、L=2、E=4、act=2、seq=256，**9 个变体全部加载成功**） | ✅ |
| `encode` / `forward` / `generate` | ✅ **能产出文本** | ✅ |
| **进程退出 / DLL 卸载** | ⚠️ **堆损坏**（`0xC0000374` / `0xC0000005`） | ✅ 退出码 0 |

**⇒ 用户能拿到正确输出，问题是"退出时崩"**，不是"算错"或"生成不了"。
之前几次测出 0/8，是因为崩溃发生在 stdout 刷新之前，我的判据（找 `GEN`）读不到输出 —— **第三次踩到"判据不可靠"的坑**。

**排序（按实际影响）**：
1. 服务端长驻进程反复 load/unload 才会累积风险（产线用 GPU 版，不受影响）；
2. `pip install` 后只用 CPU 版的脚本，退出码异常（CI 里会让构建判失败）；
3. 生成内容本身是正确的。

修法仍是那两条：`tg_load` 后做完整性校验；查退出路径的释放顺序（`tg_unload`/`DllMain`）。
