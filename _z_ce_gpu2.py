import io, py_compile, tempfile
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\python\tgai_go\train\__init__.py"
L = io.open(p, encoding="utf-8").read().split("\n")
i = next(k for k, l in enumerate(L) if l.startswith("def _cross_entropy_cuda_ignore("))
j = i + 1
while j < len(L) and not (L[j].startswith("def ") or L[j].startswith("class ") or L[j].startswith("@")):
    j += 1
new = '''def _cross_entropy_cuda_ignore(logits, tgt, ignore_index):
    """★ GPU 版（不再拷 CPU）：用 safe 目标调融合 kernel，再把无效行梯度清零。

    旧实现把整块 logits 拷到 CPU 算（10M 模型 4992x360 约 7 MB D2H + 7 MB H2D），
    实测占整步 13%。新实现全在 GPU，只把 [n,1] 的 mask 传上去。
    """
    rt = _cuda()
    n, v = logits.shape
    tgt_arr = np.asarray(tgt).reshape(-1)
    mask = (tgt_arr != ignore_index)
    nvalid = int(mask.sum())
    if nvalid == 0:
        return Tensor._from_gpu(rt.h2d(np.zeros(1, np.float32)),
                                requires_grad=logits.requires_grad, _parents=(logits,))
    safe = np.where(mask, tgt_arr, 0).astype(np.int64)
    tgt_buf = rt.h2d(safe, dtype=np.int32, cache_slot="ce_tgt")
    mean_buf = rt.ce_fwd(logits._gpu, tgt_buf, n, v)          # 对 n 求均值
    ratio = float(n) / float(nvalid)
    loss_buf = rt.scale(mean_buf, ratio) if ratio != 1.0 else mean_buf
    out = Tensor._from_gpu(loss_buf, requires_grad=logits.requires_grad, _parents=(logits,))
    mask_t = Tensor(mask.astype(np.float32).reshape(n, 1))    # [n,1]，广播到 [n,v]

    def bw(g, grads):
        if not logits.requires_grad:
            return
        scale = rt.read_scalar(g) * ratio
        dl = rt.ce_bwd(logits._gpu, tgt_buf, n, v)
        d = Tensor._from_gpu(dl) * mask_t
        if scale != 1.0:
            d = d * Tensor(np.array([float(scale)], np.float32))
        _acc(grads, logits, d._gpu)

    out._backward = bw
    return out
'''.split("\n")
L2 = L[:i] + new + L[j:]
io.open(p, "w", encoding="utf-8").write("\n".join(L2))
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)
print("已精确替换，行数 %d -> %d（语法 OK）" % (len(L), len(L2)))
