/* llama / qwen2 / gpt2 前向编排。算子全部是现成的 CUfunction / cublasSgemm_v2。
 * 架构差异只读 TgArch：qkv_bias / norm / pos / ffn / n_kv / bias_all。不复制算子树。
 *
 * LayerNorm、GELU-tanh 是共享原语：用已有的 sum_axis8 / ewise_pow / ewise_exp /
 * ewise_add / ewise_mul / ewise_scale / broadcast_to8 拼出来，三个架构都能调。
 *
 * 和 llama_ref.py / gpt2_ref.py + CudaRuntime 的发射约定对齐：
 *   2D 的 x @ W.T 走 gemm_nt（cublas，transa=T, transb=N）
 *   3D 的 bmm（RoPE 的 x@R、SDPA 的 q@kT 与 p@v）走 cublas NN，和 bmm() 一样
 *   行 bias 走 bcast_add4（[BS,N] + [N]），与 Tensor.__add__ 的广播同一套步长
 *   逐元素 / norm / embed / permute / gather / softmax / swap 走已编译 kernel
 */
#include "llama_fwd.h"

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*CuLaunchFn)(
    void* fn,
    unsigned int gx, unsigned int gy, unsigned int gz,
    unsigned int bx, unsigned int by, unsigned int bz,
    unsigned int shared, void* stream, void** params, void* extra);
typedef int (*CuH2DFn)(uint64_t dst, const void* src, size_t bytes);
typedef int (*CuD2HFn)(void* dst, uint64_t src, size_t bytes);
typedef int (*CuD2DFn)(uint64_t dst, uint64_t src, size_t bytes);
typedef int (*CuAllocFn)(uint64_t* ptr, size_t bytes);
typedef int (*CuFreeFn)(uint64_t ptr);
typedef int (*CuMemset32Fn)(uint64_t dst, unsigned int val, size_t nwords, void* stream);
typedef int (*CuStreamSyncFn)(void* stream);
typedef int (*SgemmFn)(
    void* handle, int transa, int transb,
    int m, int n, int k,
    const float* alpha,
    uint64_t A, int lda,
    uint64_t B, int ldb,
    const float* beta,
    uint64_t C, int ldc);

/* 26 个 uint64 句柄 + 7 个 int32 + TgArch(6 个 int32)。
 * 208+28+24=260，w 前补 4 字节对齐到 264。
 */
_Static_assert(sizeof(TgArch) == 24, "TgArch size");
_Static_assert(offsetof(TgaiLlamaPlan, arch) == 26 * 8 + 7 * 4, "TgArch offset");
_Static_assert(offsetof(TgaiLlamaPlan, w) == 26 * 8 + 7 * 4 + 24 + 4, "weight table offset");
_Static_assert(sizeof(TgaiLlamaPlan) == 26 * 8 + 7 * 4 + 24 + 4 + 8, "TgaiLlamaPlan size");

enum {
    W_TOK = 0,
    W_FINAL = 1,
    W_LM = 2,
    W_COS = 3,
    W_SIN = 4,
    W_R = 5,
    W_BASE = 6,
    W_PER = 9,
    W_QBIAS = 9,
    W_KBIAS = 10,
    W_VBIAS = 11,
    W_PER_BIAS = 12
};

static int g_ffi = 0;
static char g_err[512];

int tgai_llama_plan_size(void) {
    return (int)sizeof(TgaiLlamaPlan);
}

int tgai_llama_last_ffi_calls(void) {
    return g_ffi;
}

const char* tgai_llama_last_error(void) {
    return g_err;
}

static void set_err(const char* msg) {
    snprintf(g_err, sizeof(g_err), "%s", msg ? msg : "");
}

static int fail(const char* msg) {
    set_err(msg);
    return 1;
}

static int failf(const char* what, int rc) {
    snprintf(g_err, sizeof(g_err), "%s rc=%d", what, rc);
    return 1;
}

static int mul_i(int a, int b, int* out) {
    if (a < 0 || b < 0) return -1;
    if (a > 0 && b > 2147483647 / a) return -1;
    *out = a * b;
    return 0;
}

typedef struct Api {
    CuLaunchFn launch;
    CuH2DFn h2d;
    CuD2HFn d2h;
    CuD2DFn d2d;
    CuAllocFn alloc;
    CuFreeFn free;
    CuMemset32Fn memset32;
    CuStreamSyncFn sync;
    SgemmFn sgemm;
    void* cublas;
    void* k_emb;
    void* k_rms;
    void* k_perm;
    void* k_add;
    void* k_mul;
    void* k_scale;
    void* k_bcast;
    void* k_softmax;
    void* k_swap;
    void* k_silu;
    void* k_gather;
    void* k_bcast_to;
    void* k_exp;
    void* k_recip;
    void* k_pow;
    void* k_sum;
} Api;

static int bind_api(const TgaiLlamaPlan* p, Api* a) {
    memset(a, 0, sizeof(*a));
    a->launch = (CuLaunchFn)(uintptr_t)p->cu_launch;
    a->h2d = (CuH2DFn)(uintptr_t)p->cu_h2d;
    a->d2h = (CuD2HFn)(uintptr_t)p->cu_d2h;
    a->d2d = (CuD2DFn)(uintptr_t)p->cu_d2d;
    a->alloc = (CuAllocFn)(uintptr_t)p->cu_alloc;
    a->free = (CuFreeFn)(uintptr_t)p->cu_free;
    a->memset32 = (CuMemset32Fn)(uintptr_t)p->cu_memset32;
    a->sync = (CuStreamSyncFn)(uintptr_t)p->cu_stream_sync;
    a->sgemm = (SgemmFn)(uintptr_t)p->sgemm;
    a->cublas = (void*)(uintptr_t)p->cublas;
    a->k_emb = (void*)(uintptr_t)p->k_emb;
    a->k_rms = (void*)(uintptr_t)p->k_rms;
    a->k_perm = (void*)(uintptr_t)p->k_perm;
    a->k_add = (void*)(uintptr_t)p->k_add;
    a->k_mul = (void*)(uintptr_t)p->k_mul;
    a->k_scale = (void*)(uintptr_t)p->k_scale;
    a->k_bcast = (void*)(uintptr_t)p->k_bcast;
    a->k_softmax = (void*)(uintptr_t)p->k_softmax;
    a->k_swap = (void*)(uintptr_t)p->k_swap;
    a->k_silu = (void*)(uintptr_t)p->k_silu;
    a->k_gather = (void*)(uintptr_t)p->k_gather;
    a->k_bcast_to = (void*)(uintptr_t)p->k_bcast_to;
    a->k_exp = (void*)(uintptr_t)p->k_exp;
    a->k_recip = (void*)(uintptr_t)p->k_recip;
    a->k_pow = (void*)(uintptr_t)p->k_pow;
    a->k_sum = (void*)(uintptr_t)p->k_sum;
    if (!a->launch || !a->h2d || !a->d2h || !a->d2d || !a->alloc || !a->free ||
        !a->memset32 || !a->sync || !a->sgemm || !a->cublas ||
        !a->k_emb || !a->k_rms || !a->k_perm || !a->k_add || !a->k_mul ||
        !a->k_scale || !a->k_bcast || !a->k_softmax || !a->k_swap ||
        !a->k_silu || !a->k_gather || !a->k_bcast_to ||
        !a->k_exp || !a->k_recip || !a->k_pow || !a->k_sum) {
        return fail("llama plan 里有空的函数指针");
    }
    return 0;
}

static int launch(Api* a, void* fn,
                  unsigned gx, unsigned bx, unsigned shared, void** params) {
    int rc;
    if (gx < 1) gx = 1;
    if (bx < 1) bx = 1;
    rc = a->launch(fn, gx, 1, 1, bx, 1, 1, shared, NULL, params, NULL);
    if (rc != 0) return failf("cuLaunchKernel", rc);
    return 0;
}

static void grid_ewise(int n, unsigned* gx, unsigned* bx) {
    *bx = 256;
    *gx = (unsigned)((n + 255) / 256);
    if (*gx < 1) *gx = 1;
}

static int k_ewise_add(Api* a, uint64_t x, uint64_t y, uint64_t o, int n) {
    unsigned gx, bx;
    void* params[4];
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &y; params[2] = &o; params[3] = &n;
    return launch(a, a->k_add, gx, bx, 0, params);
}

static int k_ewise_mul(Api* a, uint64_t x, uint64_t y, uint64_t o, int n) {
    unsigned gx, bx;
    void* params[4];
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &y; params[2] = &o; params[3] = &n;
    return launch(a, a->k_mul, gx, bx, 0, params);
}

static int k_ewise_scale(Api* a, uint64_t x, float s, uint64_t o, int n) {
    unsigned gx, bx;
    void* params[4];
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &s; params[2] = &o; params[3] = &n;
    return launch(a, a->k_scale, gx, bx, 0, params);
}

static int k_rms(Api* a, uint64_t out, uint64_t x, uint64_t w, int seq, int d, float eps) {
    void* params[6];
    if (seq <= 0 || d <= 0) return 0;
    params[0] = &out; params[1] = &x; params[2] = &w;
    params[3] = &seq; params[4] = &d; params[5] = &eps;
    return launch(a, a->k_rms, (unsigned)seq, 256, 256 * 4, params);
}

static int k_emb(Api* a, uint64_t out, uint64_t table, uint64_t ids,
                 int seq, int d, int vocab) {
    int unk = 0;
    void* params[7];
    if (seq <= 0 || d <= 0) return 0;
    params[0] = &out; params[1] = &table; params[2] = &ids;
    params[3] = &seq; params[4] = &d; params[5] = &vocab; params[6] = &unk;
    return launch(a, a->k_emb, (unsigned)seq, 256, 0, params);
}

static int k_softmax(Api* a, uint64_t x, uint64_t y, int rows, int k) {
    unsigned gx, bx;
    void* params[4];
    if (rows <= 0 || k <= 0) return 0;
    grid_ewise(rows, &gx, &bx);
    params[0] = &x; params[1] = &y; params[2] = &rows; params[3] = &k;
    return launch(a, a->k_softmax, gx, bx, 0, params);
}

static int k_swap(Api* a, uint64_t in, uint64_t out, int batch, int m, int n) {
    unsigned gx, bx;
    int total;
    void* params[5];
    if (mul_i(batch, m, &total) || mul_i(total, n, &total)) return fail("swap 尺寸溢出");
    if (total <= 0) return 0;
    grid_ewise(total, &gx, &bx);
    params[0] = &in; params[1] = &out;
    params[2] = &batch; params[3] = &m; params[4] = &n;
    return launch(a, a->k_swap, gx, bx, 0, params);
}

static int k_silu(Api* a, uint64_t x, uint64_t ones, int n) {
    unsigned gx, bx;
    void* params[3];
    if (n <= 0) return 0;
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &ones; params[2] = &n;
    return launch(a, a->k_silu, gx, bx, 0, params);
}

static void pad8(const int* src, int n, int fill, int dst[8]) {
    int i;
    for (i = 0; i < 8; ++i) dst[i] = fill;
    for (i = 0; i < n; ++i) dst[8 - n + i] = src[i];
}

static int k_permute(Api* a, uint64_t in, uint64_t out,
                     const int* in_shape, const int* axes, int nd) {
    int in_st[8], out_shape[8], src_st[8];
    int o8[8], s8[8];
    int n = 1, i, stride = 1;
    unsigned gx, bx;
    void* params[19];
    if (nd < 1 || nd > 8) return fail("permute 维数非法");
    for (i = nd - 1; i >= 0; --i) {
        in_st[i] = stride;
        if (mul_i(stride, in_shape[i], &stride)) return fail("permute 步长溢出");
    }
    for (i = 0; i < nd; ++i) {
        int ax = axes[i];
        if (ax < 0 || ax >= nd) return fail("permute 轴非法");
        out_shape[i] = in_shape[ax];
        src_st[i] = in_st[ax];
        if (mul_i(n, out_shape[i], &n)) return fail("permute 元素溢出");
    }
    pad8(out_shape, nd, 1, o8);
    pad8(src_st, nd, 0, s8);
    if (n <= 0) return 0;
    grid_ewise(n, &gx, &bx);
    params[0] = &in; params[1] = &out; params[2] = &n;
    for (i = 0; i < 8; ++i) params[3 + i] = &o8[i];
    for (i = 0; i < 8; ++i) params[11 + i] = &s8[i];
    return launch(a, a->k_perm, gx, bx, 0, params);
}

/* 把 rank-2 的 src 按右对齐广播到 rank-3 的 out。stride 规则同 CudaRuntime.broadcast_to。 */
static int k_broadcast_to_3(Api* a, uint64_t in, uint64_t out,
                            int s0, int s1, int d0, int d1, int d2) {
    int shape[3], strides[3], o8[8], s8[8];
    int raw0, raw1, n, i;
    unsigned gx, bx;
    void* params[19];
    int aligned0 = 1, aligned1 = s0, aligned2 = s1;
    if (d0 < 1 || d1 < 1 || d2 < 1) return fail("broadcast 目标维非法");
    if (!(aligned0 == d0 || aligned0 == 1)) return fail("broadcast 维 0 对不上");
    if (!(aligned1 == d1 || aligned1 == 1)) return fail("broadcast 维 1 对不上");
    if (!(aligned2 == d2 || aligned2 == 1)) return fail("broadcast 维 2 对不上");
    raw0 = s1;
    raw1 = 1;
    shape[0] = d0; shape[1] = d1; shape[2] = d2;
    strides[0] = 0; /* i < base，base = 1 */
    strides[1] = (aligned1 == 1) ? 0 : raw0;
    strides[2] = (aligned2 == 1) ? 0 : raw1;
    if (mul_i(d0, d1, &n) || mul_i(n, d2, &n)) return fail("broadcast 元素溢出");
    pad8(shape, 3, 1, o8);
    pad8(strides, 3, 0, s8);
    grid_ewise(n, &gx, &bx);
    params[0] = &in; params[1] = &out; params[2] = &n;
    for (i = 0; i < 8; ++i) params[3 + i] = &o8[i];
    for (i = 0; i < 8; ++i) params[11 + i] = &s8[i];
    return launch(a, a->k_bcast_to, gx, bx, 0, params);
}

/* repeat_interleave(dim=0)：源 [lead, S, DH] → [lead*rep, S, DH]。 */
static int k_repeat0(Api* a, uint64_t in, uint64_t out,
                     int lead, int S, int DH, int rep) {
    int oshape[3], divs[3], st[3];
    int o8[8], d8[8], s8[8];
    int n, base = 0, i;
    unsigned gx, bx;
    void* params[28];
    if (rep == 1) return fail("repeat0 不该在 rep==1 时发射");
    if (lead < 1 || S < 1 || DH < 1 || rep < 1) return fail("repeat 尺寸非法");
    if (lead > 2147483647 / rep) return fail("repeat lead 溢出");
    oshape[0] = lead * rep;
    oshape[1] = S;
    oshape[2] = DH;
    divs[0] = rep; divs[1] = 1; divs[2] = 1;
    st[2] = 1;
    st[1] = DH;
    if (mul_i(S, DH, &st[0])) return fail("repeat 步长溢出");
    if (mul_i(oshape[0], S, &n) || mul_i(n, DH, &n)) return fail("repeat 元素溢出");
    pad8(oshape, 3, 1, o8);
    pad8(divs, 3, 1, d8);
    pad8(st, 3, 0, s8);
    grid_ewise(n, &gx, &bx);
    params[0] = &in; params[1] = &out; params[2] = &n; params[3] = &base;
    for (i = 0; i < 8; ++i) params[4 + i] = &o8[i];
    for (i = 0; i < 8; ++i) params[12 + i] = &d8[i];
    for (i = 0; i < 8; ++i) params[20 + i] = &s8[i];
    return launch(a, a->k_gather, gx, bx, 0, params);
}

/* scores[BH,S,S] + mask[S,S]。步长和 CudaRuntime.add_broadcast / _strides4 一致。 */
static int k_add_mask(Api* a, uint64_t scores, uint64_t mask, uint64_t out,
                      int BH, int S) {
    int cpad[4], spad[4], mpad[4], sa[4], sb[4];
    int n, i;
    unsigned gx, bx;
    void* params[17];
    if (BH < 1 || S < 1) return fail("mask 尺寸非法");
    if (mul_i(BH, S, &n) || mul_i(n, S, &n)) return fail("mask 元素溢出");
    cpad[0] = 1; cpad[1] = BH; cpad[2] = S; cpad[3] = S;
    spad[0] = 1; spad[1] = BH; spad[2] = S; spad[3] = S;
    mpad[0] = 1; mpad[1] = 1; mpad[2] = S; mpad[3] = S;
    {
        int stride = 1;
        for (i = 3; i >= 0; --i) {
            sa[i] = (spad[i] == 1) ? 0 : stride;
            stride *= spad[i];
        }
        stride = 1;
        for (i = 3; i >= 0; --i) {
            sb[i] = (mpad[i] == 1) ? 0 : stride;
            stride *= mpad[i];
        }
    }
    grid_ewise(n, &gx, &bx);
    params[0] = &scores; params[1] = &mask; params[2] = &out; params[3] = &n;
    params[4] = &cpad[0]; params[5] = &cpad[1]; params[6] = &cpad[2]; params[7] = &cpad[3];
    params[8] = &sa[0]; params[9] = &sa[1]; params[10] = &sa[2]; params[11] = &sa[3];
    params[12] = &sb[0]; params[13] = &sb[1]; params[14] = &sb[2]; params[15] = &sb[3];
    (void)params[16];
    return launch(a, a->k_bcast, gx, bx, 0, params);
}

/* out[rows, n] = x[rows, n] + bias[n]。步长和 CudaRuntime.add_broadcast 的 2D+1D 相同。
 * x 与 out 可以是同一块：每个下标只读自己再写自己。
 */
static int k_add_row_bias(Api* a, uint64_t x, uint64_t bias, uint64_t out, int rows, int n) {
    int total;
    unsigned gx, bx;
    int c0 = 1, c1 = 1, c2, c3;
    int a0 = 0, a1 = 0, a2, a3 = 1;
    int b0 = 0, b1 = 0, b2 = 0, b3 = 1;
    void* params[16];
    if (rows < 1 || n < 1) return fail("bias 尺寸非法");
    if (!bias) return fail("qkv bias 指针为空");
    if (mul_i(rows, n, &total)) return fail("bias 元素溢出");
    c2 = rows;
    c3 = n;
    a2 = n;
    grid_ewise(total, &gx, &bx);
    params[0] = &x; params[1] = &bias; params[2] = &out; params[3] = &total;
    params[4] = &c0; params[5] = &c1; params[6] = &c2; params[7] = &c3;
    params[8] = &a0; params[9] = &a1; params[10] = &a2; params[11] = &a3;
    params[12] = &b0; params[13] = &b1; params[14] = &b2; params[15] = &b3;
    return launch(a, a->k_bcast, gx, bx, 0, params);
}

static int k_exp(Api* a, uint64_t x, uint64_t o, int n) {
    unsigned gx, bx;
    void* params[3];
    if (n <= 0) return 0;
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &o; params[2] = &n;
    return launch(a, a->k_exp, gx, bx, 0, params);
}

static int k_pow(Api* a, uint64_t x, float p, uint64_t o, int n) {
    unsigned gx, bx;
    void* params[4];
    if (n <= 0) return 0;
    grid_ewise(n, &gx, &bx);
    params[0] = &x; params[1] = &p; params[2] = &o; params[3] = &n;
    return launch(a, a->k_pow, gx, bx, 0, params);
}

/* broadcast_to8。shape/strides 是左填充到 8 维之后的输出形状和源步长。 */
static int k_broadcast8(Api* a, uint64_t in, uint64_t out, int n,
                        const int* shape, const int* strides) {
    int sh[8], st[8], i;
    unsigned gx, bx;
    void* params[19];
    if (n <= 0) return 0;
    for (i = 0; i < 8; ++i) {
        sh[i] = shape[i];
        st[i] = strides[i];
    }
    grid_ewise(n, &gx, &bx);
    params[0] = &in; params[1] = &out; params[2] = &n;
    for (i = 0; i < 8; ++i) params[3 + i] = &sh[i];
    for (i = 0; i < 8; ++i) params[11 + i] = &st[i];
    return launch(a, a->k_bcast_to, gx, bx, 0, params);
}

/* [cols] → [rows, cols]。 */
static int k_bcast_vec(Api* a, uint64_t vec, uint64_t out, int rows, int cols) {
    int shape[8], strides[8], n, i;
    if (rows < 1 || cols < 1) return fail("bcast vec 尺寸非法");
    if (mul_i(rows, cols, &n)) return fail("bcast vec 溢出");
    for (i = 0; i < 8; ++i) {
        shape[i] = 1;
        strides[i] = 0;
    }
    shape[6] = rows;
    shape[7] = cols;
    strides[7] = 1;
    return k_broadcast8(a, vec, out, n, shape, strides);
}

/* [rows, 1] 连续 → [rows, cols]。源的每一行一个标量。 */
static int k_bcast_col(Api* a, uint64_t col, uint64_t out, int rows, int cols) {
    int shape[8], strides[8], n, i;
    if (rows < 1 || cols < 1) return fail("bcast col 尺寸非法");
    if (mul_i(rows, cols, &n)) return fail("bcast col 溢出");
    for (i = 0; i < 8; ++i) {
        shape[i] = 1;
        strides[i] = 0;
    }
    shape[6] = rows;
    shape[7] = cols;
    strides[6] = 1;
    return k_broadcast8(a, col, out, n, shape, strides);
}

/* 沿最后一维求和。[rows, cols] → [rows]，和 sum_axis8 keepdims 的内存布局一致。 */
static int k_sum_last(Api* a, uint64_t in, uint64_t out, int rows, int cols) {
    int shp[8], i;
    int n_out, axis, extent;
    unsigned gx, bx;
    void* params[13];
    if (rows <= 0 || cols <= 0) return 0;
    for (i = 0; i < 6; ++i) shp[i] = 1;
    shp[6] = rows;
    shp[7] = cols;
    n_out = rows;
    axis = 7;
    extent = cols;
    grid_ewise(n_out, &gx, &bx);
    params[0] = &in; params[1] = &out;
    params[2] = &n_out; params[3] = &axis; params[4] = &extent;
    for (i = 0; i < 8; ++i) params[5 + i] = &shp[i];
    return launch(a, a->k_sum, gx, bx, 0, params);
}

static unsigned float_bits(float x) {
    unsigned u = 0;
    memcpy(&u, &x, sizeof(u));
    return u;
}

/* LayerNorm(eps)。x/out 是 [rows, d]，gamma/beta 是 [d]。
 * tmp/tmp2 各 rows*d，red/red2 各 rows。公式和 gpt2_ref._layernorm 一样：
 * mean、中心化、var、rsqrt、乘 gamma、加 beta。除的是 d，不是 d-1。
 */
static int layernorm(Api* a, uint64_t out, uint64_t x, uint64_t gamma, uint64_t beta,
                     uint64_t red, uint64_t red2, uint64_t tmp, uint64_t tmp2,
                     int rows, int d, float eps) {
    int n = 0;
    float inv_d;
    if (!gamma || !beta) return fail("layernorm 缺少 gamma 或 beta");
    if (rows < 1 || d < 1) return fail("layernorm 尺寸非法");
    if (mul_i(rows, d, &n)) return fail("layernorm 元素溢出");
    inv_d = (float)(1.0 / (double)d);
    if (k_sum_last(a, x, red, rows, d)) return 1;
    if (k_ewise_scale(a, red, inv_d, red, rows)) return 1;
    if (k_bcast_col(a, red, tmp, rows, d)) return 1;
    if (k_ewise_scale(a, tmp, -1.f, tmp, n)) return 1;
    if (k_ewise_add(a, x, tmp, tmp2, n)) return 1;
    if (k_ewise_mul(a, tmp2, tmp2, tmp, n)) return 1;
    if (k_sum_last(a, tmp, red, rows, d)) return 1;
    if (k_ewise_scale(a, red, inv_d, red, rows)) return 1;
    if (a->memset32(red2, float_bits(eps), (size_t)rows, NULL) != 0)
        return fail("cuMemsetD32Async(layernorm eps)");
    if (k_ewise_add(a, red, red2, red, rows)) return 1;
    if (k_pow(a, red, -0.5f, red, rows)) return 1;
    if (k_bcast_col(a, red, tmp, rows, d)) return 1;
    if (k_ewise_mul(a, tmp2, tmp, tmp, n)) return 1;
    if (k_bcast_vec(a, gamma, tmp2, rows, d)) return 1;
    if (k_ewise_mul(a, tmp, tmp2, tmp, n)) return 1;
    if (k_bcast_vec(a, beta, tmp2, rows, d)) return 1;
    if (k_ewise_add(a, tmp, tmp2, out, n)) return 1;
    return 0;
}

/* GPT-2 gelu_new：0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))。
 * tanh(z)=2*sigmoid(2z)-1，sigmoid(z)=1/(1+exp(-z))，倒数用 pow(-1)，
 * 和 gpt2_ref.gelu_tanh / _sigmoid 同一串算子。ones 必须是 n 个 1。
 * y 可以和 t1/t2 重叠，不能和 x、ones 重叠。
 */
static int gelu_tanh(Api* a, uint64_t x, uint64_t y, uint64_t t1, uint64_t t2,
                     uint64_t ones, int n) {
    const float gelu_c = (float)sqrt(2.0 / 3.14159265358979323846);
    const float gelu_k = (float)0.044715;
    if (n <= 0) return 0;
    if (!x || !y || !t1 || !t2 || !ones) return fail("gelu 缓冲为空");
    if (k_ewise_mul(a, x, x, t1, n)) return 1;
    if (k_ewise_mul(a, t1, x, t2, n)) return 1;
    if (k_ewise_scale(a, t2, gelu_k, t2, n)) return 1;
    if (k_ewise_add(a, x, t2, t1, n)) return 1;
    if (k_ewise_scale(a, t1, gelu_c, t1, n)) return 1;
    if (k_ewise_scale(a, t1, 2.f, t1, n)) return 1;
    if (k_ewise_scale(a, t1, -1.f, t1, n)) return 1;
    if (k_exp(a, t1, t2, n)) return 1;
    if (k_ewise_add(a, t2, ones, t2, n)) return 1;
    if (k_pow(a, t2, -1.f, t2, n)) return 1;
    if (k_ewise_scale(a, t2, 2.f, t2, n)) return 1;
    if (k_ewise_scale(a, ones, -1.f, t1, n)) return 1;
    if (k_ewise_add(a, t2, t1, t2, n)) return 1;
    if (k_ewise_add(a, t2, ones, t1, n)) return 1;
    if (k_ewise_mul(a, x, t1, y, n)) return 1;
    if (k_ewise_scale(a, y, 0.5f, y, n)) return 1;
    return 0;
}

/* 行主序 C[M,N] = A[M,K] @ B[K,N]。与 CudaRuntime._sgemm NN 同一组参数。 */
static int gemm_nn(Api* a, uint64_t A, uint64_t B, uint64_t C, int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;
    int rc;
    if (M == 0 || N == 0 || K == 0) return 0;
    rc = a->sgemm(a->cublas, 0, 0, N, M, K, &alpha, B, N, A, K, &beta, C, N);
    if (rc != 0) return failf("cublasSgemm_v2 NN", rc);
    return 0;
}

/* 行主序 C[M,N] = A[M,K] @ B[N,K]^T。与 gemm_nt 同一组参数。 */
static int gemm_nt(Api* a, uint64_t A, uint64_t B, uint64_t C, int M, int N, int K) {
    float alpha = 1.f, beta = 0.f;
    int rc;
    if (M == 0 || N == 0 || K == 0) return 0;
    rc = a->sgemm(a->cublas, 1, 0, N, M, K, &alpha, B, K, A, K, &beta, C, N);
    if (rc != 0) return failf("cublasSgemm_v2 NT", rc);
    return 0;
}

static int bmm_nn(Api* a, uint64_t A, uint64_t B, uint64_t C,
                  int batch, int M, int N, int K) {
    int b, a_n, b_n, c_n;
    if (mul_i(M, K, &a_n) || mul_i(K, N, &b_n) || mul_i(M, N, &c_n))
        return fail("bmm 尺寸溢出");
    for (b = 0; b < batch; ++b) {
        uint64_t As = A + (uint64_t)b * (uint64_t)a_n * 4u;
        uint64_t Bs = B + (uint64_t)b * (uint64_t)b_n * 4u;
        uint64_t Cs = C + (uint64_t)b * (uint64_t)c_n * 4u;
        if (gemm_nn(a, As, Bs, Cs, M, N, K)) return 1;
    }
    return 0;
}

/* x[BH,S,DH] @ R[DH,DH]，R 每个 batch 同一块。 */
static int bmm_right_shared(Api* a, uint64_t X, uint64_t R, uint64_t C,
                            int batch, int S, int DH) {
    int b, xs;
    if (mul_i(S, DH, &xs)) return fail("rope gemm 溢出");
    for (b = 0; b < batch; ++b) {
        uint64_t Xs = X + (uint64_t)b * (uint64_t)xs * 4u;
        uint64_t Cs = C + (uint64_t)b * (uint64_t)xs * 4u;
        if (gemm_nn(a, Xs, R, Cs, S, DH, DH)) return 1;
    }
    return 0;
}

typedef struct Scratch {
    int B, S, D, NH, NKV, FF, V;
    uint64_t base;
    size_t cap;
    uint64_t ids, mask, ones;
    uint64_t red, red2;
    uint64_t h, h2, nrm, q, k, v, qh, kh, vh, rot;
    uint64_t cos_b, sin_b, t1, t2;
    uint64_t k_rep, v_rep, kt, scores, scores2, attn, o_lin, proj, n2;
    uint64_t gate, up, silu, gu, down, hn, logits;
} Scratch;

static Scratch g_sc;
static CuFreeFn g_free;

static void scratch_reset(void) {
    if (g_sc.base && g_free) g_free(g_sc.base);
    memset(&g_sc, 0, sizeof(g_sc));
}

static uint64_t bump(uint64_t* cursor, size_t* used, size_t cap, int nfloat) {
    size_t bytes, aligned;
    uint64_t p;
    if (nfloat < 0) return 0;
    bytes = (size_t)nfloat * 4u;
    aligned = (bytes + 255u) & ~255u;
    if (*used > cap || aligned > cap - *used) return 0;
    p = *cursor + *used;
    *used += aligned;
    return p;
}

static int scratch_fit(Api* a, int B, int S, int D, int DH, int NH, int NKV,
                       int FF, int V, Scratch* sc) {
    int BS, BH, BKV, kv_dim;
    size_t need = 0, used = 0;
    uint64_t cursor;
    int counts[40];
    int i, nc = 0;
    uint64_t* slots[40];
    int ns = 0;

    if (mul_i(B, S, &BS)) return fail("B*S 溢出");
    if (mul_i(B, NH, &BH)) return fail("B*NH 溢出");
    if (mul_i(B, NKV, &BKV)) return fail("B*NKV 溢出");
    if (mul_i(NKV, DH, &kv_dim)) return fail("kv 维溢出");

    memset(sc, 0, sizeof(*sc));
    {
        int ones_n = 0, mask_n = 0, logits_n = 0, scores_n = 0;
        int h_n = 0, k_n = 0, kv_s = 0, q_n = 0;
        if (S > 0 && mul_i(S, S, &mask_n)) return fail("mask 溢出");
        if (mul_i(BS, FF, &ones_n)) return fail("ones 溢出");
        if (mul_i(BS, D, &h_n)) return fail("h 溢出");
        if (mul_i(BS, kv_dim, &k_n)) return fail("k 溢出");
        if (mul_i(BKV, S, &kv_s) || mul_i(kv_s, DH, &kv_s)) return fail("kh 溢出");
        if (mul_i(BH, S, &q_n) || mul_i(q_n, DH, &q_n)) return fail("qh 溢出");
        if (mul_i(BH, S, &scores_n) || mul_i(scores_n, S, &scores_n)) return fail("scores 溢出");
        if (mul_i(BS, V, &logits_n)) return fail("logits 溢出");

#define PUSH(field, count) do { counts[nc++] = (count); slots[ns++] = &sc->field; } while (0)
        PUSH(ids, BS);
        PUSH(mask, mask_n);
        PUSH(ones, ones_n);
        PUSH(red, BS);
        PUSH(red2, BS);
        PUSH(h, h_n);
        PUSH(h2, h_n);
        PUSH(nrm, h_n);
        PUSH(q, h_n);
        PUSH(k, k_n);
        PUSH(v, k_n);
        PUSH(qh, q_n);
        PUSH(kh, kv_s);
        PUSH(vh, kv_s);
        PUSH(rot, q_n);
        PUSH(cos_b, q_n);
        PUSH(sin_b, q_n);
        PUSH(t1, q_n);
        PUSH(t2, q_n);
        PUSH(k_rep, q_n);
        PUSH(v_rep, q_n);
        PUSH(kt, q_n);
        PUSH(scores, scores_n);
        PUSH(scores2, scores_n);
        PUSH(attn, q_n);
        PUSH(o_lin, h_n);
        PUSH(proj, h_n);
        PUSH(n2, h_n);
        PUSH(gate, ones_n);
        PUSH(up, ones_n);
        PUSH(silu, ones_n);
        PUSH(gu, ones_n);
        PUSH(down, h_n);
        PUSH(hn, h_n);
        PUSH(logits, logits_n);
#undef PUSH
    }

    for (i = 0; i < nc; ++i) {
        size_t bytes = (size_t)counts[i] * 4u;
        size_t aligned = (bytes + 255u) & ~255u;
        if (need > (size_t)-1 - aligned) return fail("scratch 总字节溢出");
        need += aligned;
    }
    if (need < 256) need = 256;
    if (g_sc.base && g_sc.cap >= need && g_sc.B == B && g_sc.S == S &&
        g_sc.D == D && g_sc.NH == NH && g_sc.NKV == NKV &&
        g_sc.FF == FF && g_sc.V == V) {
        *sc = g_sc;
        return 0;
    }
    scratch_reset();
    g_free = a->free;
    {
        uint64_t ptr = 0;
        int rc = a->alloc(&ptr, need);
        if (rc != 0 || !ptr) return failf("cuMemAlloc(scratch)", rc);
        g_sc.base = ptr;
        g_sc.cap = need;
        g_sc.B = B;
        g_sc.S = S;
        g_sc.D = D;
        g_sc.NH = NH;
        g_sc.NKV = NKV;
        g_sc.FF = FF;
        g_sc.V = V;
    }
    cursor = g_sc.base;
    used = 0;
    for (i = 0; i < ns; ++i) {
        uint64_t p = bump(&cursor, &used, need, counts[i]);
        if (!p && counts[i] > 0) {
            scratch_reset();
            return fail("scratch 切分失败");
        }
        /* 0 个元素也给一个对齐地址，调用方不得发射。 */
        if (!p) p = cursor + used;
        *slots[i] = p;
    }
    sc->base = g_sc.base;
    sc->cap = g_sc.cap;
    sc->B = B;
    sc->S = S;
    sc->D = D;
    sc->NH = NH;
    sc->NKV = NKV;
    sc->FF = FF;
    sc->V = V;
    g_sc = *sc;
    return 0;
}

static int rope(Api* a, uint64_t x, uint64_t out, Scratch* sc,
                uint64_t cos, uint64_t sin, uint64_t R,
                int BH, int S, int DH) {
    int n;
    if (mul_i(BH, S, &n) || mul_i(n, DH, &n)) return fail("rope 元素溢出");
    if (k_broadcast_to_3(a, cos, sc->cos_b, S, DH, BH, S, DH)) return 1;
    if (k_broadcast_to_3(a, sin, sc->sin_b, S, DH, BH, S, DH)) return 1;
    if (bmm_right_shared(a, x, R, sc->rot, BH, S, DH)) return 1;
    if (k_ewise_mul(a, x, sc->cos_b, sc->t1, n)) return 1;
    if (k_ewise_mul(a, sc->rot, sc->sin_b, sc->t2, n)) return 1;
    if (k_ewise_add(a, sc->t1, sc->t2, out, n)) return 1;
    return 0;
}

static int apply_norm(Api* a, int kind, uint64_t out, uint64_t x,
                      uint64_t gamma, uint64_t beta, Scratch* sc, int seq, int d) {
    switch (kind) {
    case TG_NORM_RMS:
        return k_rms(a, out, x, gamma, seq, d, (float)1e-6);
    case TG_NORM_LN:
        /* q / proj 在三次调用点都已经用完或还没写入，借来当 rows*d 的临时区。 */
        return layernorm(a, out, x, gamma, beta, sc->red, sc->red2, sc->q, sc->proj,
                         seq, d, (float)1e-5);
    default:
        return fail("norm 未实现");
    }
}

static int apply_pos(Api* a, int kind, uint64_t x, uint64_t out, Scratch* sc,
                     uint64_t cos, uint64_t sin, uint64_t R,
                     int BH, int S, int DH) {
    switch (kind) {
    case TG_POS_ROPE:
        return rope(a, x, out, sc, cos, sin, R, BH, S, DH);
    default:
        return fail("pos 只实现了 rope");
    }
}

/* bias 指针为 0 就不发射。llama / qwen2 走这条时三个 bias 都是 0，内核序列和改前一样。 */
static int apply_ffn(Api* a, int kind,
                     uint64_t gate_w, uint64_t up_w, uint64_t down_w,
                     uint64_t gate_b, uint64_t up_b, uint64_t down_b,
                     Scratch* sc, int BS, int D, int FF) {
    int f_n = 0;
    int h_n = 0;
    if (mul_i(BS, FF, &f_n)) return fail("ffn 溢出");
    if (mul_i(BS, D, &h_n)) return fail("ffn 残差溢出");
    switch (kind) {
    case TG_FFN_SWIGLU:
        if (gemm_nt(a, sc->n2, gate_w, sc->gate, BS, FF, D)) return 1;
        if (gate_b && k_add_row_bias(a, sc->gate, gate_b, sc->gate, BS, FF)) return 1;
        if (gemm_nt(a, sc->n2, up_w, sc->up, BS, FF, D)) return 1;
        if (up_b && k_add_row_bias(a, sc->up, up_b, sc->up, BS, FF)) return 1;
        if (a->d2d(sc->silu, sc->gate, (size_t)f_n * 4u) != 0)
            return fail("cuMemcpyDtoD(silu)");
        if (k_silu(a, sc->silu, sc->ones, f_n)) return 1;
        if (k_ewise_mul(a, sc->silu, sc->up, sc->gu, f_n)) return 1;
        if (gemm_nt(a, sc->gu, down_w, sc->down, BS, D, FF)) return 1;
        if (down_b && k_add_row_bias(a, sc->down, down_b, sc->down, BS, D)) return 1;
        if (k_ewise_add(a, sc->h2, sc->down, sc->h, h_n)) return 1;
        return 0;
    case TG_FFN_GELU:
        /* fc 写到 gate，GELU 写到 up，proj 写到 down。ones 仍是全 1。 */
        if (gemm_nt(a, sc->n2, gate_w, sc->gate, BS, FF, D)) return 1;
        if (gate_b && k_add_row_bias(a, sc->gate, gate_b, sc->gate, BS, FF)) return 1;
        if (gelu_tanh(a, sc->gate, sc->up, sc->silu, sc->gu, sc->ones, f_n)) return 1;
        if (gemm_nt(a, sc->up, down_w, sc->down, BS, D, FF)) return 1;
        if (down_b && k_add_row_bias(a, sc->down, down_b, sc->down, BS, D)) return 1;
        if (k_ewise_add(a, sc->h2, sc->down, sc->h, h_n)) return 1;
        return 0;
    default:
        return fail("ffn 未实现");
    }
}

static int sdpa(Api* a, uint64_t q, uint64_t k, uint64_t v, uint64_t o,
                Scratch* sc, int BH, int S, int DH, float scale) {
    int scores_n, rows;
    if (mul_i(BH, S, &rows) || mul_i(rows, S, &scores_n)) return fail("sdpa scores 溢出");
    if (k_swap(a, k, sc->kt, BH, S, DH)) return 1;
    if (bmm_nn(a, q, sc->kt, sc->scores, BH, S, S, DH)) return 1;
    if (k_ewise_scale(a, sc->scores, scale, sc->scores, scores_n)) return 1;
    if (k_add_mask(a, sc->scores, sc->mask, sc->scores2, BH, S)) return 1;
    if (k_softmax(a, sc->scores2, sc->scores, rows, S)) return 1;
    if (bmm_nn(a, sc->scores, v, o, BH, S, DH, S)) return 1;
    return 0;
}

typedef struct LayerW {
    uint64_t n1, n1b, q, k, v, o, n2, n2b;
    uint64_t gate, up, down;
    uint64_t qb, kb, vb, ob, gateb, upb, downb;
} LayerW;

/* 头和每层的条数。llama = 6 + 9L，qwen2 = 6 + 12L，gpt2 = 5 + 16L。 */
static int weight_layout(const TgArch* arch, int* head, int* per) {
    int h = 3;
    int p = 0;
    if (arch->qkv_bias != 0 && arch->qkv_bias != 1) return fail("qkv_bias 只能是 0 或 1");
    if (arch->bias_all != 0 && arch->bias_all != 1) return fail("bias_all 只能是 0 或 1");
    if (arch->pos == TG_POS_ROPE) h += 3;
    else if (arch->pos == TG_POS_WPE) h += 1;
    else return fail("pos 未实现");
    if (arch->norm == TG_NORM_LN) h += 1;
    else if (arch->norm != TG_NORM_RMS) return fail("norm 未实现");
    p += (arch->norm == TG_NORM_LN) ? 2 : 1;
    p += 4;
    p += (arch->norm == TG_NORM_LN) ? 2 : 1;
    if (arch->ffn == TG_FFN_SWIGLU) p += 3;
    else if (arch->ffn == TG_FFN_GELU) p += 2;
    else return fail("ffn 未实现");
    if (arch->qkv_bias || arch->bias_all) p += 3;
    if (arch->bias_all) {
        p += 1;
        p += (arch->ffn == TG_FFN_SWIGLU) ? 3 : 2;
    }
    *head = h;
    *per = p;
    return 0;
}

static int decode_layer(const uint64_t* lw, const TgArch* arch, LayerW* o) {
    int p = 0;
    memset(o, 0, sizeof(*o));
    o->n1 = lw[p++];
    if (arch->norm == TG_NORM_LN) o->n1b = lw[p++];
    o->q = lw[p++];
    o->k = lw[p++];
    o->v = lw[p++];
    o->o = lw[p++];
    o->n2 = lw[p++];
    if (arch->norm == TG_NORM_LN) o->n2b = lw[p++];
    if (arch->ffn == TG_FFN_SWIGLU) {
        o->gate = lw[p++];
        o->up = lw[p++];
        o->down = lw[p++];
    } else {
        o->gate = lw[p++];
        o->down = lw[p++];
    }
    if (arch->qkv_bias || arch->bias_all) {
        o->qb = lw[p++];
        o->kb = lw[p++];
        o->vb = lw[p++];
    }
    if (arch->bias_all) {
        o->ob = lw[p++];
        if (arch->ffn == TG_FFN_SWIGLU) {
            o->gateb = lw[p++];
            o->upb = lw[p++];
            o->downb = lw[p++];
        } else {
            o->gateb = lw[p++];
            o->downb = lw[p++];
        }
    }
    return p;
}

int tgai_llama_forward(
    const TgaiLlamaPlan* plan,
    const int64_t* ids,
    int32_t B,
    int32_t S,
    float* logits) {
    Api api;
    Scratch sc;
    const uint64_t* w;
    int D, L, NH, NKV, FF, V, DH, n_rep, layer, per, head;
    int BS, kv_dim;
    float scale;
    int32_t* ids32 = NULL;
    float* mask = NULL;
    int rc_sync;
    const int ax_q[4] = {0, 2, 1, 3};

    g_ffi = 1;
    g_err[0] = 0;
    if (!plan || !logits) return fail("plan/logits 为空");
    if (B < 0 || S < 0) return fail("B/S 为负");
    if (B == 0 || S == 0) return 0;
    if (!ids) return fail("ids 为空");
    if (bind_api(plan, &api)) return 1;

    D = plan->d;
    L = plan->L;
    NH = plan->nh;
    NKV = plan->arch.n_kv;
    FF = plan->ff;
    V = plan->V;
    if (weight_layout(&plan->arch, &head, &per)) return 1;
    if (D <= 0 || L < 0 || NH <= 0 || NKV <= 0 || FF <= 0 || V <= 0)
        return fail("cfg 维度非法");
    if (D % NH) return fail("d 不能被 nh 整除");
    if (NH % NKV) return fail("nh 不能被 n_kv 整除");
    DH = D / NH;
    if (DH <= 0) return fail("head_dim 非法");
    if (plan->arch.pos == TG_POS_ROPE && (DH & 1)) return fail("head_dim 必须是正偶数");
    n_rep = NH / NKV;
    if (S > plan->rope_seq) return fail("S 超过位置表的 seq");
    if (L > 0 && per > 0 && L > 2147483647 / per) return fail("权重条数溢出");
    if (head > 2147483647 - L * per) return fail("权重条数溢出");
    if (plan->n_w != head + L * per) return fail("权重条数和描述符对不上");
    w = (const uint64_t*)(uintptr_t)plan->w;
    if (!w) return fail("权重表为空");

    if (mul_i(B, S, &BS)) return fail("B*S 溢出");
    if (mul_i(NKV, DH, &kv_dim)) return fail("kv_dim 溢出");
    scale = (float)(1.0 / sqrt((double)DH));

    if (scratch_fit(&api, B, S, D, DH, NH, NKV, FF, V, &sc)) return 1;

    ids32 = (int32_t*)malloc((size_t)BS * sizeof(int32_t));
    mask = (float*)malloc((size_t)S * (size_t)S * sizeof(float));
    if (!ids32 || !mask) {
        free(ids32);
        free(mask);
        return fail("host scratch malloc 失败");
    }
    {
        int i, j;
        for (i = 0; i < BS; ++i) {
            int64_t id = ids[i];
            if (id < 0 || id >= V) {
                free(ids32);
                free(mask);
                return fail("Embedding 下标越界");
            }
            ids32[i] = (int32_t)id;
        }
        for (i = 0; i < S; ++i)
            for (j = 0; j < S; ++j)
                mask[i * S + j] = (j > i) ? (float)-1.0e30 : 0.f;
    }
    if (api.h2d(sc.ids, ids32, (size_t)BS * 4u) != 0) {
        free(ids32); free(mask);
        return fail("cuMemcpyHtoD(ids)");
    }
    if (api.h2d(sc.mask, mask, (size_t)S * (size_t)S * 4u) != 0) {
        free(ids32); free(mask);
        return fail("cuMemcpyHtoD(mask)");
    }
    free(ids32);
    free(mask);
    {
        int ones_n = 0;
        if (mul_i(BS, FF, &ones_n)) return fail("ones 溢出");
        if (api.memset32(sc.ones, 0x3f800000u, (size_t)ones_n, NULL) != 0)
            return fail("cuMemsetD32Async(ones)");
    }

    if (k_emb(&api, sc.h, w[W_TOK], sc.ids, BS, D, V)) return 1;
    if (plan->arch.pos == TG_POS_WPE) {
        int h_n = 0;
        if (mul_i(BS, D, &h_n)) return fail("wpe 溢出");
        /* wpe 在头的第 4 项（tok、gamma、lm 之后）。只取前 S 行。 */
        if (k_broadcast_to_3(&api, w[3], sc.proj, S, D, B, S, D)) return 1;
        if (k_ewise_add(&api, sc.h, sc.proj, sc.h, h_n)) return 1;
    }

    for (layer = 0; layer < L; ++layer) {
        const uint64_t* lw = w + head + layer * per;
        LayerW lwts;
        int q_shape[4], k_shape[4];
        int o_shape[4];
        int h_n = 0, q_n = 0;
        uint64_t k_use, v_use;
        if (decode_layer(lw, &plan->arch, &lwts) != per)
            return fail("层权重步长不一致");
        if (mul_i(BS, D, &h_n)) return fail("h_n 溢出");
        if (mul_i(B * NH, S, &q_n) || mul_i(q_n, DH, &q_n)) return fail("q_n 溢出");

        if (apply_norm(&api, plan->arch.norm, sc.nrm, sc.h, lwts.n1, lwts.n1b, &sc, BS, D))
            return 1;
        if (gemm_nt(&api, sc.nrm, lwts.q, sc.q, BS, D, D)) return 1;
        if (gemm_nt(&api, sc.nrm, lwts.k, sc.k, BS, kv_dim, D)) return 1;
        if (gemm_nt(&api, sc.nrm, lwts.v, sc.v, BS, kv_dim, D)) return 1;
        if (lwts.qb) {
            if (k_add_row_bias(&api, sc.q, lwts.qb, sc.q, BS, D)) return 1;
            if (k_add_row_bias(&api, sc.k, lwts.kb, sc.k, BS, kv_dim)) return 1;
            if (k_add_row_bias(&api, sc.v, lwts.vb, sc.v, BS, kv_dim)) return 1;
        }

        q_shape[0] = B; q_shape[1] = S; q_shape[2] = NH; q_shape[3] = DH;
        k_shape[0] = B; k_shape[1] = S; k_shape[2] = NKV; k_shape[3] = DH;
        if (k_permute(&api, sc.q, sc.qh, q_shape, ax_q, 4)) return 1;
        if (k_permute(&api, sc.k, sc.kh, k_shape, ax_q, 4)) return 1;
        if (k_permute(&api, sc.v, sc.vh, k_shape, ax_q, 4)) return 1;

        if (plan->arch.pos == TG_POS_ROPE) {
            if (apply_pos(&api, plan->arch.pos, sc.qh, sc.qh, &sc,
                          w[W_COS], w[W_SIN], w[W_R], B * NH, S, DH))
                return 1;
            if (apply_pos(&api, plan->arch.pos, sc.kh, sc.kh, &sc,
                          w[W_COS], w[W_SIN], w[W_R], B * NKV, S, DH))
                return 1;
        }

        if (n_rep == 1) {
            k_use = sc.kh;
            v_use = sc.vh;
        } else {
            if (k_repeat0(&api, sc.kh, sc.k_rep, B * NKV, S, DH, n_rep)) return 1;
            if (k_repeat0(&api, sc.vh, sc.v_rep, B * NKV, S, DH, n_rep)) return 1;
            k_use = sc.k_rep;
            v_use = sc.v_rep;
        }

        if (sdpa(&api, sc.qh, k_use, v_use, sc.attn, &sc, B * NH, S, DH, scale))
            return 1;

        o_shape[0] = B; o_shape[1] = NH; o_shape[2] = S; o_shape[3] = DH;
        if (k_permute(&api, sc.attn, sc.o_lin, o_shape, ax_q, 4)) return 1;
        if (gemm_nt(&api, sc.o_lin, lwts.o, sc.proj, BS, D, D)) return 1;
        if (lwts.ob && k_add_row_bias(&api, sc.proj, lwts.ob, sc.proj, BS, D)) return 1;
        if (k_ewise_add(&api, sc.h, sc.proj, sc.h2, h_n)) return 1;

        if (apply_norm(&api, plan->arch.norm, sc.n2, sc.h2, lwts.n2, lwts.n2b, &sc, BS, D))
            return 1;
        if (apply_ffn(&api, plan->arch.ffn, lwts.gate, lwts.up, lwts.down,
                      lwts.gateb, lwts.upb, lwts.downb, &sc, BS, D, FF))
            return 1;
    }

    {
        int h_n = 0, logits_n = 0;
        uint64_t final_b = 0;
        int final_b_at = 3;
        if (plan->arch.pos == TG_POS_ROPE) final_b_at += 3;
        else final_b_at += 1;
        if (plan->arch.norm == TG_NORM_LN) final_b = w[final_b_at];
        if (mul_i(BS, D, &h_n)) return fail("final h 溢出");
        if (mul_i(BS, V, &logits_n)) return fail("logits 溢出");
        if (apply_norm(&api, plan->arch.norm, sc.hn, sc.h, w[W_FINAL], final_b, &sc, BS, D))
            return 1;
        if (gemm_nt(&api, sc.hn, w[W_LM], sc.logits, BS, V, D)) return 1;
        rc_sync = api.sync(NULL);
        if (rc_sync != 0) return failf("cuStreamSynchronize", rc_sync);
        if (api.d2h(logits, sc.logits, (size_t)logits_n * 4u) != 0)
            return fail("cuMemcpyDtoH(logits)");
    }
    set_err("");
    return 0;
}
