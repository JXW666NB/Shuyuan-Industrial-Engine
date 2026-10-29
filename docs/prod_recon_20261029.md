# 产线侦察（2026-10-29）：推理服务在哪、用什么、怎么回滚

> 用户已授权改动产线（"反正没人用"）。本文先记录事实与回滚方案，**动手前**必须先读。

## 1. 产线在哪

- 主机：**V100 那台**（`Administrator@192.168.0.103`，即 `v100.py` 连的机器）
- 监听：`0.0.0.0:11450`，PID **4684**
- 命令行：
  ```
  C:\Python311\python.exe -u D:\TGAI\02_数据\run_tgaigo_server.py
    -m D:\TGAI\models\V2.7\TGAI_NB_V2.7.tg
    --host 0.0.0.0 --port 11450 --think-level off
    --dll D:\TGAI\libtgai_engine_gpu.dll
  ```
- 健康检查：`http://192.168.0.103:11450/health` → `{"status":"ok","model":"tgai","requests":1018,...}`

**注意**：产线用的是 **GPU 版引擎**（`libtgai_engine_gpu.dll`），
而我复现内存安全问题的本地测试用的是**CPU 版**（`tgai_go/_native/libtgai_engine.dll`）。
两者不是同一个二进制，修的时候要分清。

## 2. 源码与工具链

| 位置 | 内容 |
|---|---|
| `D:\tgai_nlp\TGAI GO\src\tgai_engine.c` | CPU 引擎主源（232 KB） |
| 同目录 `tgai_gpu_bridge.c` / `tgai_gpu_dispatch.c` | GPU 桥接/派发 |
| `D:\tgai_nlp\TGAI GO\rebuild_cpu.bat` | CPU 重编脚本 |
| `D:\tgai_nlp\TGAI GO\build_cuda.bat` / `build_cuda_v100.bat` | GPU 重编脚本 |
| `D:\tgai_nlp\TGAI GO\tcc\` | 随仓库的 Tiny C Compiler |
| `C:\...\Python312\Lib\site-packages\tgai_go\_native\libtgai_engine.dll` | 已安装的 CPU 引擎（292 KB，2026-10-18） |

## 3. 要修的两个缺陷（本轮已定性）

1. **退出阶段堆损坏**：进程结束时报 `0xC0000374`（HEAP_CORRUPTION）。
   所有文件都中，包括仓库自带样例 ⇒ 与"加载/生成成功与否"无关，是**释放路径**的问题。
2. **运行阶段访问违例**：`0xC0000005`，我们的合成 `.TG` 约 20–40% 触发，
   仓库样例不触发 ⇒ 与文件内容相关，触发点未定位。

影响：**任何一份"结构合法但内容异常"的 `.TG` 都可能让推理进程崩掉**。
生产上等于"用户上传一个坏模型就能把服务打挂"。

## 4. 回滚方案（动手前先做）

1. 备份 `D:\TGAI\libtgai_engine_gpu.dll`（或 CPU 版）到带时间戳的副本；
2. 记录 PID 与命令行（本文 §1 已记录，另存一份到产线机器的文本）；
3. 重启方式：结束 PID 4684，再用 §1 的完整命令行拉起来
   （**不要用 taskkill /f**，先确认能否温和停止；该服务是否有看护/自启任务需再确认）；
4. 回滚 = 把备份的 DLL 覆盖回去 + 按同样方式重启。

## 5. 下一步（按顺序）

1. 在 V100 上备份 DLL 与启动脚本，并确认"停/起"的安全方式；
2. 用 `rebuild_cpu.bat`（tcc）在**本地**先重编 CPU 引擎，确认能编过（不动产线）；
3. 在 CPU 引擎上加最小防护（token id / 权重指针的范围检查），用我已有的复现脚本验证：
   目标 = 坏模型**不再崩进程**，而是返回错误；好模型行为不变；
4. CPU 版验证通过后，再决定是否把同样的修改搬到 GPU 引擎（`build_cuda_v100.bat`）。
