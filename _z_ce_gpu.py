import io, py_compile, tempfile
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\python\tgai_go\train\__init__.py"
s = io.open(p, encoding="utf-8").read()
i = s.index("def _cross_entropy_cuda_ignore(")
j = s.index("def _cross_entropy_cuda_ignore_end_marker", i) if "def _cross_entropy_cuda_ignore_end_marker" in s else None
# 找到该函数结束：下一个顶层 def
k = s.index("\ndef ", i + 10) + 1
old = s[i:k]
if "★ GPU 版" in old:
    print("已是新版，跳过")
else:
    new = '''def _cross_entropy_cuda_ignore(logits, tgt, ignore_index):
    """★ GPU 版（不再拷 CPU）：用 safe 目标调融合 kernel，再把无效行的梯度清零。

    旧实现把整块 logits 拷到 CPU 算（10M 模型 4992×360 约 7 MB D2H + 7 MB H2D），
    实测占整步 13%（8.8 ms / 66 ms）。新实现在 GPU 上完成，只把 [n,1] 的 mask 传上去。
    """
    rt = _cuda()
    n, v = logits.shape
    tgt = np.asarray(tgt).reshape(-1)
    mask = (tgt != ignore_index)
    nvalid = int(mask.sum())
    if nvalid == 0:
        zero = Tensor._from_gpu(rt.h2d(np.zeros(1, np.float32)),
                                requires_grad=logits.requires_grad, _parents=(logits,))
        return zero
    safe = np.where(mask, tgt, 0).astype(np.int64)
    tgt_buf = rt.h2d(safe, dtype=np.int32, cache_slot="ce_tgt")
    mean_buf = rt.ce_fwd(logits._gpu, tgt_buf, n, v)      # 对 n 求均值
    ratio = float(n) / float(nvalid)
    loss_buf = rt.scale(mean_buf, ratio) if ratio != 1.0 else mean_buf
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))

    mask_t = Tensor(mask.astype(np.float32).reshape(n, 1))   # [n,1] 广播

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = rt.read_scalar(g) * ratio
        dl = rt.ce_bwd(logits._gpu, tgt_buf, n, v)
        d = Tensor._from_gpu(dl) * mask_t                     # 无效行清零
        if scale != 1.0:
            d = d * Tensor(np.array([float(scale)], np.float32))
        _acc(grads, logits, d._gpu)

    out._backward = bw
    return out


'''
    s = s[:i] + new + s[k:]
    io.open(p, "w", encoding="utf-8").write(s)
    py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)
    print("已替换为 GPU 版 CE-ignore（语法 OK）")
