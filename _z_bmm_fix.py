import io
p = r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\python\tgai_go\train\cuda.py"
s = io.open(p, encoding="utf-8").read()

anchor = "    def softmax_lastdim(self, x):"
if "def sgemm_strided_batched" not in s:
    new_method = '''    def sgemm_strided_batched(self, A, B, batch, M, K, N):
        """★ 一次 cuBLAS 调用算完整个 batch（行优先 C[b] = A[b] @ B[b]）。

        以前的 bmm 是在 Python 里 for i in range(batch): _sgemm(...)，
        attention 里 batch = B*NH = 624 ⇒ 每次 bmm 要 624 次 ctypes 调用（约 12 ms）。
        换成 cublasSgemmStridedBatched 后是 1 次调用。
        """
        C = self.empty((batch, M, N), dtype=np.float32)
        if batch == 0 or M == 0 or N == 0:
            return C
        if K == 0:
            return self.zeros((batch, M, N), dtype=np.float32)
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        cb = self.cublas
        cb.cublasSgemmStridedBatched.argtypes = [
            ctypes.c_void_p, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p, ctypes.c_int, ctypes.c_longlong,
            ctypes.c_int,
        ]
        cb.cublasSgemmStridedBatched.restype = ctypes.c_int
        self._mark_submitted(A.ptr)
        self._mark_submitted(B.ptr)
        self._mark_submitted(C.ptr)
        self._activate()
        # 行优先 → 列优先：交换 A/B 与其 leading dim（与 _sgemm 同一套转换）
        self._chk_blas(cb.cublasSgemmStridedBatched(
            self.handle, CUDA_OP_N, CUDA_OP_N,
            int(N), int(M), int(K),
            ctypes.byref(alpha),
            ctypes.c_void_p(int(B.ptr)), int(N), int(K * N),
            ctypes.c_void_p(int(A.ptr)), int(K), int(M * K),
            ctypes.byref(beta),
            ctypes.c_void_p(int(C.ptr)), int(N), int(M * N),
            int(batch),
        ), "cublasSgemmStridedBatched")
        return C

'''
    assert anchor in s
    s = s.replace(anchor, new_method + anchor, 1)

old_bmm_tail = """        self._mark_submitted(A.ptr)
        self._mark_submitted(B.ptr)
        self._mark_submitted(C.ptr)
        for i in range(batch):
            self._sgemm(
                CUDA_OP_N, CUDA_OP_N, N, M, K,
                B.ptr + i * b_bytes, N,
                A.ptr + i * a_bytes, K,
                C.ptr + i * c_bytes, N,
            )
        return C"""
new_bmm_tail = """        # ★ 批量走 cuBLAS strided-batched（1 次调用），小 batch 才回退到循环
        if batch >= 4:
            return self.sgemm_strided_batched(A, B, batch, M, K, N)
        self._mark_submitted(A.ptr)
        self._mark_submitted(B.ptr)
        self._mark_submitted(C.ptr)
        for i in range(batch):
            self._sgemm(
                CUDA_OP_N, CUDA_OP_N, N, M, K,
                B.ptr + i * b_bytes, N,
                A.ptr + i * a_bytes, K,
                C.ptr + i * c_bytes, N,
            )
        return C"""
assert old_bmm_tail in s
s = s.replace(old_bmm_tail, new_bmm_tail, 1)
io.open(p, "w", encoding="utf-8").write(s)
import py_compile, tempfile
py_compile.compile(p, cfile=tempfile.mktemp(), doraise=True)
print("已加 sgemm_strided_batched 并让 bmm 走它（语法 OK）")
