# -*- coding: utf-8 -*-
"""数据并行（多卡）地基：参数广播 + 梯度全归约。

设计目标：**先正确，再快**。第一版走「本机共享内存」做交换，
不依赖 NCCL / MPI / torch.distributed —— 我们本来就是自研运行时。

进程模型：每个 rank 一个进程（各自的 CUDA_VISIBLE_DEVICES），
通过 `TGAI_DP_WORLD` / `TGAI_DP_RANK` 环境变量认领身份，
顺次目录（rendezvous）默认 `/tmp/tgai_dp_<job>`，Windows 下用 `%TEMP%`。

用法：
    dp = DataParallel()                    # 从环境变量读 world/rank
    dp.broadcast(params)                   # rank0 的权重分发给所有人（保证起点一致）
    ... 前向/反向 ...
    dp.all_reduce_grads(params)            # 梯度求平均，使每个 rank 拿到相同的更新方向
    opt.step()

world=1 时所有操作都是**快速空操作**，所以单卡跑不会有任何行为变化。
"""
import os
import tempfile
import time

import numpy as np

try:
    from multiprocessing import shared_memory
    _HAS_SHM = True
except Exception:                                    # noqa: BLE001
    _HAS_SHM = False


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


class DataParallel:
    def __init__(self, world=None, rank=None, job=None, timeout=600.0):
        self.world = _env_int("TGAI_DP_WORLD", 1) if world is None else int(world)
        self.rank = _env_int("TGAI_DP_RANK", 0) if rank is None else int(rank)
        self.job = job or os.environ.get("TGAI_DP_JOB", "default")
        self.timeout = float(timeout)
        base = os.environ.get("TGAI_DP_DIR") or tempfile.gettempdir()
        self.dir = os.path.join(base, "tgai_dp_%s" % self.job)
        self._shm = []
        if self.world > 1 and not _HAS_SHM:
            raise RuntimeError("多卡需要 multiprocessing.shared_memory，当前环境不可用")
        if self.world > 1:
            os.makedirs(self.dir, exist_ok=True)
            self._slot = None
            self._bytes = 0

    # ---------- 基础设施 ----------

    def _marker(self, name):
        return os.path.join(self.dir, "%s.%d" % (name, self.rank))

    def _wait_for(self, path, what):
        t0 = time.time()
        while not os.path.exists(path):
            if time.time() - t0 > self.timeout:
                raise TimeoutError("数据并行等待超时（%s）: %s" % (what, path))
            time.sleep(0.002)

    def _barrier(self, tag):
        """简易 barrier：每个 rank 落一个文件，等所有 rank 到齐。

        ★ 必须带「世代号」且**不删除**标记文件：
          早期版本里 rank0 通过后立刻删掉自己的标记，rank1 可能还没看到就已经被删，
          于是永久等待（实测 2 进程死锁）。带上自增世代之后，每次 barrier 用全新文件名，
          旧文件留着也不会让下一轮误判为「已到齐」。
        """
        self._bar_gen = int(getattr(self, "_bar_gen", 0)) + 1
        t = "%s_%d" % (tag, self._bar_gen)
        p = os.path.join(self.dir, "bar_%s.%d" % (t, self.rank))
        with open(p, "wb") as f:
            f.write(b"1")
        for r in range(self.world):
            self._wait_for(os.path.join(self.dir, "bar_%s.%d" % (t, r)), "barrier " + t)

    def _ensure_slot(self, nbytes):
        """所有 rank 共用一块 `world * nbytes` 的共享内存，按 rank 分段写。"""
        if self._slot is not None and self._bytes >= nbytes:
            return
        if self._slot is not None:
            self._slot.close()
        need = int(nbytes)
        if self.rank == 0:
            self._slot = shared_memory.SharedMemory(
                name="tgai_dp_%s_%d" % (self.job, need), create=True, size=need * self.world)
            self._slot.buf[:8] = need.to_bytes(8, "little")
            self._shm.append(self._slot)
        else:
            name = "tgai_dp_%s_%d" % (self.job, need)
            t0 = time.time()
            while True:
                try:
                    self._slot = shared_memory.SharedMemory(name=name, create=False)
                    break
                except FileNotFoundError:
                    if time.time() - t0 > self.timeout:
                        raise TimeoutError("共享内存 %s 未创建" % name)
                    time.sleep(0.002)
            self._shm.append(self._slot)
        self._bytes = need

    # ---------- 对外接口 ----------

    def broadcast(self, params):
        """把 rank0 的权重分发给所有 rank（保证各卡起点完全一致）。"""
        if self.world <= 1:
            return
        arrays = [np.ascontiguousarray(v.data if hasattr(v, "data") else v, dtype=np.float32)
                  for v in params]
        total = sum(a.size for a in arrays)
        self._ensure_slot(max(total * 4, 8))
        off = 0
        if self.rank == 0:
            for a in arrays:
                self._slot.buf[off:off + a.nbytes] = a.tobytes()
                off += a.nbytes
        self._barrier("bcast")
        if self.rank != 0:
            off = 0
            for a in arrays:
                a[...] = np.frombuffer(bytes(self._slot.buf[off:off + a.nbytes]),
                                       dtype=np.float32).reshape(a.shape)
                off += a.nbytes
        self._barrier("bcast2")

    def all_reduce_grads(self, params, scale=None):
        """梯度全归约（默认求平均）。params 里每个元素要有 .grad（numpy 数组）。"""
        if self.world <= 1:
            return
        grads = []
        for p in params:
            g = getattr(p, "grad", None)
            if g is None:
                continue
            arr = g if isinstance(g, np.ndarray) else np.asarray(g, dtype=np.float32)
            grads.append((p, arr))
        if not grads:
            return
        total = sum(arr.size for _p, arr in grads)
        self._ensure_slot(max(total * 4, 8))
        off = 0
        for _p, arr in grads:                       # 各 rank 写自己那一段
            b = np.ascontiguousarray(arr, dtype=np.float32).tobytes()
            self._slot.buf[self.rank * total * 4 + off:self.rank * total * 4 + off + len(b)] = b
            off += len(b)
        self._barrier("grads")
        buf = np.ndarray((self.world, total), dtype=np.float32,
                         buffer=self._slot.buf, offset=0)
        acc = buf.sum(axis=0)
        if scale is None:
            acc /= float(self.world)
        else:
            acc *= float(scale)
        off = 0
        for p, arr in grads:
            n = arr.size
            p.grad = acc[off:off + n].astype(arr.dtype, copy=True).reshape(arr.shape)
            off += n

    def close(self):
        try:
            for f in os.listdir(self.dir):
                if f.startswith("bar_") and f.endswith(".%d" % self.rank):
                    os.remove(os.path.join(self.dir, f))
        except OSError:
            pass
        for s in self._shm:
            try:
                s.close()
            except Exception:                        # noqa: BLE001
                pass
            try:
                if self.rank == 0:
                    s.unlink()
            except Exception:                        # noqa: BLE001
                pass
        self._shm = []

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False
