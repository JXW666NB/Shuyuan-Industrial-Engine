# -*- coding: utf-8 -*-
"""tgai_go.train 的 CUDA 运行时（M1 + Linux + M2 + M3 + M5 算子）。

不 import torch。只通过 ctypes 调用驱动、cuBLAS、NVRTC：

    Windows: nvcuda.dll / cublas64_*.dll / nvrtc64_*.dll
    Linux:   libcuda.so.1 / libcublas.so.12 / libnvrtc.so.12
             （cuBLAS、NVRTC 用完整路径 CDLL，不改 LD_LIBRARY_PATH）

内嵌 device 源码来自 training/kernels/tgai_ce.cu、tgai_fwd.cu、tgai_bwd.cu，
编译进本包，训练包不 import training 目录。算子公式不在这里重写。

cuBLAS 句柄设为 CUBLAS_PEDANTIC_MATH，让 cublasSgemm 走
CUBLAS_COMPUTE_32F_PEDANTIC，避免 Ampere+ 默认 TF32 把 fp32 GEMM 拉到 ~1e-3。
"""
from __future__ import annotations

import ctypes
import os
import sys

import numpy as np

CUDA_OP_N = 0
CUDA_OP_T = 1
# cublas_api.h: CUBLAS_PEDANTIC_MATH = 2
# 注释写明：cublas<T> 例程因此改用对应的 _PEDANTIC compute type。
CUBLAS_PEDANTIC_MATH = 2
CUBLAS_POINTER_MODE_HOST = 0

# cuda.h
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR = 75
CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR = 76

_CE_AND_HELPERS = r"""
/* transplanted from training/kernels/tgai_ce.cu — keep the math identical. */

__device__ float exp32_bits(float x);
__device__ float log32_bits(float x);

/* 一行一个 block、块内 256 线程。发射端必须用这个块大小。
 * 每线程扫一段连续列，块内再按线程号左折叠。V<=256 时每段至多一个元素，
 * 和原来的单线程左折叠逐位相同。exp32 / log32 / __fadd_rn 不换。
 * sm 由 kernel 的 __shared__ 传入；归约函数末尾再栅栏一次，避免下一次复用 sm 时读穿。
 */
__device__ float ce_red_max(float v, float* sm) {
    sm[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float m = sm[0];
        for (int i = 1; i < 256; ++i) m = fmaxf(m, sm[i]);
        sm[0] = m;
    }
    __syncthreads();
    float out = sm[0];
    __syncthreads();
    return out;
}

__device__ float ce_red_sum(float v, float* sm) {
    sm[threadIdx.x] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.0f;
        for (int i = 0; i < 256; ++i) s = __fadd_rn(s, sm[i]);
        sm[0] = s;
    }
    __syncthreads();
    float out = sm[0];
    __syncthreads();
    return out;
}

__device__ void ce_chunk_range(int V, int* begin_out, int* end_out) {
    const int tid = (int)threadIdx.x;
    const int chunk = (V + 255) >> 8;
    int begin = tid * chunk;
    int end = begin + chunk;
    if (begin > V) begin = V;
    if (end > V) end = V;
    *begin_out = begin;
    *end_out = end;
}

__device__ void ce_row_stats(const float* row, int V, float* sm, float* m_out, float* sum_out) {
    int begin, end;
    ce_chunk_range(V, &begin, &end);
    float m = __int_as_float(0xff800000u);
    for (int j = begin; j < end; ++j) m = fmaxf(m, row[j]);
    m = ce_red_max(m, sm);
    float s = 0.0f;
    for (int j = begin; j < end; ++j) {
        float e = exp32_bits(__fsub_rn(row[j], m));
        s = __fadd_rn(s, e);
    }
    s = ce_red_sum(s, sm);
    *m_out = m;
    *sum_out = s;
}

extern "C" __global__ void ce_fwd_rows(const float* logits, const int* targets, float* terms,
                                  int N, int V) {
    __shared__ float sm[256];
    const int i = (int)blockIdx.x;
    const float* row = logits + static_cast<long long>(i) * (long long)V;
    float m, sum;
    ce_row_stats(row, V, sm, &m, &sum);
    if (threadIdx.x == 0) {
        const int t = targets[i];
        terms[i] = __fsub_rn(__fadd_rn(m, log32_bits(sum)), row[t]);
    }
}

extern "C" __global__ void ce_fwd_mean(const float* terms, float* loss, int N) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float acc = 0.0f;
    for (int i = 0; i < N; ++i) acc = __fadd_rn(acc, terms[i]);
    loss[0] = __fdiv_rn(acc, static_cast<float>(N));
}

extern "C" __global__ void ce_bwd(const float* logits, const int* targets, float* dlogits,
                                  int N, int V) {
    __shared__ float sm[256];
    const int i = (int)blockIdx.x;
    const float* row = logits + static_cast<long long>(i) * (long long)V;
    float* grow = dlogits + static_cast<long long>(i) * (long long)V;
    float m, sum;
    ce_row_stats(row, V, sm, &m, &sum);
    const float inv_sum = __fdiv_rn(1.0f, sum);
    const float inv_n = __fdiv_rn(1.0f, static_cast<float>(N));
    const int t = targets[i];
    int begin, end;
    ce_chunk_range(V, &begin, &end);
    for (int j = begin; j < end; ++j) {
        float e = exp32_bits(__fsub_rn(row[j], m));
        float p = __fmul_rn(e, inv_sum);
        if (j == t) p = __fsub_rn(p, 1.0f);
        grow[j] = __fmul_rn(p, inv_n);
    }
}

extern "C" __global__ void transpose2d(const float* a, float* b, int rows, int cols) {
    int n = rows * cols;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        int r = i / cols;
        int c = i - r * cols;
        b[c * rows + r] = a[i];
    }
}

extern "C" __global__ void ewise_add(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

extern "C" __global__ void ewise_scale(const float* a, float s, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * s;
}

extern "C" __global__ void ewise_mul(const float* a, const float* b, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] * b[i];
}

/* 与 Python _exp32 / _log32 同一套 float32 运算。不用 expf/logf，也不用 SVML。 */
__device__ float exp32_taylor(float x) {
    float cs[13];
    cs[0] = __int_as_float(0x310f76c8);
    cs[1] = __int_as_float(0x32d7322c);
    cs[2] = __int_as_float(0x3493f27e);
    cs[3] = __int_as_float(0x3638ef1d);
    cs[4] = __int_as_float(0x37d00d01);
    cs[5] = __int_as_float(0x39500d01);
    cs[6] = __int_as_float(0x3ab60b61);
    cs[7] = __int_as_float(0x3c088889);
    cs[8] = __int_as_float(0x3d2aaaab);
    cs[9] = __int_as_float(0x3e2aaaab);
    cs[10] = 0.5f;
    cs[11] = 1.0f;
    cs[12] = 1.0f;
    float p = cs[0];
    for (int i = 1; i < 13; ++i) p = __fadd_rn(__fmul_rn(p, x), cs[i]);
    return p;
}

__device__ void exp32_pair(float x, float* y_out, float* dy_out) {
    if (!(x == x)) {
        *y_out = x;
        *dy_out = x;
        return;
    }
    if (fabsf(x) <= __int_as_float(0x40200000)) {
        *y_out = exp32_taylor(x);
        *dy_out = *y_out;
        return;
    }
    if (x > 80.0f) x = 80.0f;
    if (x < -80.0f) x = -80.0f;
    float log2e = __int_as_float(0x3fb8aa3b);
    float ln2 = __int_as_float(0x3f317218);
    float nf = rintf(__fmul_rn(x, log2e));
    float t = __fsub_rn(x, __fmul_rn(nf, ln2));
    float cs[8];
    cs[0] = __int_as_float(0x39500d01);
    cs[1] = __int_as_float(0x3ab60b61);
    cs[2] = __int_as_float(0x3c088889);
    cs[3] = __int_as_float(0x3d2aaaab);
    cs[4] = __int_as_float(0x3e2aaaab);
    cs[5] = 0.5f;
    cs[6] = 1.0f;
    cs[7] = 1.0f;
    float p = cs[0];
    float dp = 0.0f;
    for (int i = 1; i < 8; ++i) {
        dp = __fadd_rn(__fmul_rn(dp, t), p);
        p = __fadd_rn(__fmul_rn(p, t), cs[i]);
    }
    int n = (int)nf;
    int one_bits = __float_as_int(1.0f);
    float scale = __int_as_float(one_bits + n * 8388608);
    *y_out = __int_as_float(__float_as_int(p) + n * 8388608);
    *dy_out = __fmul_rn(dp, scale);
}

__device__ float exp32_bits(float x) {
    float y, dy;
    exp32_pair(x, &y, &dy);
    return y;
}

__device__ float log32_bits(float x) {
    unsigned ub = __float_as_uint(x);
    int e = (int)((ub >> 23) & 255u) - 127;
    float m = __uint_as_float((ub & 0x7fffffu) | 0x3f800000u);
    float num = __fsub_rn(m, 1.0f);
    float den = __fadd_rn(m, 1.0f);
    float z = __fdiv_rn(num, den);
    float z2 = __fmul_rn(z, z);
    float z3 = __fmul_rn(z2, z);
    float z5 = __fmul_rn(z3, z2);
    float z7 = __fmul_rn(z5, z2);
    float z9 = __fmul_rn(z7, z2);
    float s = z;
    s = __fadd_rn(s, __fmul_rn(z3, __int_as_float(0x3eaaaaab)));
    s = __fadd_rn(s, __fmul_rn(z5, __int_as_float(0x3e4ccccd)));
    s = __fadd_rn(s, __fmul_rn(z7, __int_as_float(0x3e124925)));
    s = __fadd_rn(s, __fmul_rn(z9, __int_as_float(0x3de38e39)));
    float ln_m = __fmul_rn(2.0f, s);
    return __fadd_rn(ln_m, __fmul_rn((float)e, __int_as_float(0x3f317218)));
}

extern "C" __global__ void ewise_exp32(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = exp32_bits(a[i]);
}

extern "C" __global__ void ewise_log32(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = log32_bits(a[i]);
}

extern "C" __global__ void ewise_exp(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = expf(a[i]);
}

extern "C" __global__ void ewise_log(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = logf(a[i]);
}

extern "C" __global__ void ewise_recip(const float* a, float* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = __fdiv_rn(1.f, a[i]);
}

/* One block, 256 threads. The launch site must use that block size. */
extern "C" __global__ void reduce_sum(const float* a, float* out, int n) {
    __shared__ float smem[256];
    float local = 0.f;
    for (int i = (int)threadIdx.x; i < n; i += (int)blockDim.x) {
        local += a[i];
    }
    smem[threadIdx.x] = local;
    __syncthreads();
    for (int s = (int)blockDim.x >> 1; s > 0; s >>= 1) {
        if ((int)threadIdx.x < s) smem[threadIdx.x] += smem[threadIdx.x + s];
        __syncthreads();
    }
    if (threadIdx.x == 0) out[0] = smem[0];
}
"""

# 下面几段从 training/kernels/tgai_fwd.cu、tgai_bwd.cu 原样摘出来。
# 不改公式。silu_mul_fwd 是原地的，Python 侧先 clone 再调。
_M2_KERNELS = r"""
extern "C" __global__ void rmsnorm_fwd(float* out, const float* x, const float* w,
                                       int seq, int d, float eps) {
    const int t = (int)blockIdx.x;
    if (t >= seq || d <= 0) return;

    const float* row = x + (long long)t * d;
    float* orow = out + (long long)t * d;

    float local = 0.f;
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) {
        float v = row[i];
        local += v * v;
    }

    extern __shared__ float red[];
    red[threadIdx.x] = local;
    __syncthreads();
    for (int stride = (int)blockDim.x >> 1; stride > 0; stride >>= 1) {
        if ((int)threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }

    float inv = rsqrtf(red[0] / (float)d + eps);
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) {
        orow[i] = row[i] * inv * w[i];
    }
}

__device__ __forceinline__ void silu32_pair(float v, float* y_out, float* d_out) {
    float neg = __fmul_rn(v, -1.0f);
    float s, ds_dneg;
    exp32_pair(neg, &s, &ds_dneg);
    (void)ds_dneg;
    float ds = __fmul_rn(s, -1.0f);
    float den = __fadd_rn(s, 1.0f);
    float inv = __fdiv_rn(1.0f, den);
    float y = __fmul_rn(v, inv);
    float ninv = __fmul_rn(__fmul_rn(-1.0f, inv), inv);
    float dinv = __fmul_rn(ninv, ds);
    *y_out = y;
    *d_out = __fadd_rn(inv, __fmul_rn(v, dinv));
}

extern "C" __global__ void silu_mul_fwd(float* a, const float* b, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < n) {
        float y, d;
        silu32_pair(a[i], &y, &d);
        a[i] = __fmul_rn(y, b[i]);
    }
}

extern "C" __global__ void emb_gather_fwd(float* out, const float* emb, const int* ids,
                                          int seq, int d, int vocab, int unk) {
    int t = (int)blockIdx.x;
    if (t >= seq) return;
    int id = ids[t];
    if (id < 0 || id >= vocab) id = unk;
    const float* src = emb + (long long)id * d;
    float* dst = out + (long long)t * d;
    for (int i = (int)threadIdx.x; i < d; i += (int)blockDim.x) dst[i] = src[i];
}

__device__ __forceinline__ float blk_sum(float v, float* smem) {
    const int tid = threadIdx.x;
    smem[tid] = v;
    __syncthreads();
    for (int s = blockDim.x >> 1; s > 0; s >>= 1) {
        if (tid < s)
            smem[tid] += smem[tid + s];
        __syncthreads();
    }
    const float out = smem[0];
    __syncthreads();
    return out;
}

extern "C" __global__ void rmsnorm_bwd(const float* dy, const float* x, const float* w,
                                       float* dx, float* dw, int seq, int d, float eps) {
    __shared__ float smem[256];
    const int t = (int)blockIdx.x;
    if (t >= seq)
        return;

    const int tid = (int)threadIdx.x;
    const int stride = (int)blockDim.x;
    const float* xrow = x + t * d;
    const float* dyrow = dy + t * d;
    float* dxrow = dx + t * d;

    float sumsq = 0.f;
    float dot = 0.f;
    for (int i = tid; i < d; i += stride) {
        const float xi = xrow[i];
        sumsq += xi * xi;
        dot += dyrow[i] * xi * w[i];
    }
    sumsq = blk_sum(sumsq, smem);
    dot = blk_sum(dot, smem);

    const float inv_d = 1.f / (float)d;
    const float rstd = rsqrtf(sumsq * inv_d + eps);
    const float scale = rstd * rstd * rstd * inv_d;

    for (int i = tid; i < d; i += stride) {
        const float xi = xrow[i];
        const float dyi = dyrow[i];
        dxrow[i] = rstd * dyi * w[i] - scale * xi * dot;
        atomicAdd(dw + i, dyi * xi * rstd);
    }
}

extern "C" __global__ void silu_mul_bwd(const float* g, const float* a, const float* b,
                                        float* da, float* db, int n) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n)
        return;
    const float av = a[i];
    float y, dsilu;
    silu32_pair(av, &y, &dsilu);
    const float gv = g[i];
    da[i] = __fmul_rn(__fmul_rn(gv, b[i]), dsilu);
    db[i] = __fmul_rn(gv, y);
}

extern "C" __global__ void emb_gather_bwd(const float* g, const int* ids, float* demb,
                                         int seq, int d) {
    const int t = (int)blockIdx.x;
    if (t >= seq)
        return;
    const int id = ids[t];
    const float* grow = g + t * d;
    float* erow = demb + id * d;
    for (int j = (int)threadIdx.x; j < d; j += (int)blockDim.x)
        atomicAdd(erow + j, grow[j]);
}
"""


# M3：reshape 只改形状（主机 memcpy）。下面是 attention 公式里 CPU 还在用、
# 但设备上还没有的几步。dh=64 的 flash/tiled kernel 对判分台的 Dh=8 会直接 return，
# 所以 SDPA 走和 CPU 相同的 bmm + 掩码 + softmax，不改那两份已验收源码。
_M3_KERNELS = r"""
extern "C" __global__ void swap_last2(const float* a, float* b, int batch, int m, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = batch * n * m;
    if (i >= total || m <= 0 || n <= 0) return;
    int m_i = i % m;
    int n_i = (i / m) % n;
    int b_i = i / (m * n);
    b[i] = a[(b_i * m + m_i) * n + n_i];
}

/* 4D broadcast add. c0..c3 is the output shape (leading dims padded to 1).
   a0..a3 and b0..b3 are source strides; a broadcast axis has stride 0.
   Caller guarantees every output extent is >= 1. */
extern "C" __global__ void bcast_add4(
    const float* a, const float* b, float* c, int n,
    int c0, int c1, int c2, int c3,
    int a0, int a1, int a2, int a3,
    int b0, int b1, int b2, int b3) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    int t = i;
    int i3 = t % c3; t /= c3;
    int i2 = t % c2; t /= c2;
    int i1 = t % c1; t /= c1;
    int i0 = t;
    int ia = i0 * a0 + i1 * a1 + i2 * a2 + i3 * a3;
    int ib = i0 * b0 + i1 * b1 + i2 * b2 + i3 * b3;
    c[i] = a[ia] + b[ib];
}

extern "C" __global__ void softmax_lastdim(const float* x, float* y, int rows, int k) {
    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= rows || k <= 0) return;
    const float* row = x + (long long)r * k;
    float* orow = y + (long long)r * k;
    float m = row[0];
    for (int j = 1; j < k; ++j) m = fmaxf(m, row[j]);
    float sum = 0.f;
    for (int j = 0; j < k; ++j) {
        float e = expf(row[j] - m);
        orow[j] = e;
        sum += e;
    }
    float inv = (sum > 0.f) ? (1.f / sum) : 0.f;
    for (int j = 0; j < k; ++j) orow[j] *= inv;
}

extern "C" __global__ void softmax_lastdim_bwd(const float* g, const float* y, float* dx,
                                              int rows, int k) {
    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= rows || k <= 0) return;
    const float* grow = g + (long long)r * k;
    const float* yrow = y + (long long)r * k;
    float* drow = dx + (long long)r * k;
    float dot = 0.f;
    for (int j = 0; j < k; ++j) dot += grow[j] * yrow[j];
    for (int j = 0; j < k; ++j) drow[j] = yrow[j] * (grow[j] - dot);
}
"""


# M5：permute / 按轴求和（广播的反向）/ 广播成目标形状 / 最后一维 log_softmax / AdamW。
# 形状和步长在主机上算（纯标量）。张量值不回主机。
_M5_KERNELS = r"""
extern "C" __global__ void permute8(
    const float* in, float* out, int n,
    int o0, int o1, int o2, int o3, int o4, int o5, int o6, int o7,
    int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    int shp[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
    int st[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    int rem = i;
    int src = 0;
    for (int d = 7; d >= 0; --d) {
        int sd = shp[d];
        int coord = rem % sd;
        rem /= sd;
        src += coord * st[d];
    }
    out[i] = in[src];
}

extern "C" __global__ void broadcast_to8(
    const float* in, float* out, int n,
    int o0, int o1, int o2, int o3, int o4, int o5, int o6, int o7,
    int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    int shp[8] = {o0, o1, o2, o3, o4, o5, o6, o7};
    int st[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    int rem = i;
    int src = 0;
    for (int d = 7; d >= 0; --d) {
        int sd = shp[d];
        int coord = rem % sd;
        rem /= sd;
        src += coord * st[d];
    }
    out[i] = in[src];
}

/* 沿 axis 求和，输出按该维长度视为 1 的行主序写下（keepdims 布局）。 */
extern "C" __global__ void sum_axis8(
    const float* in, float* out, int n_out, int axis, int extent,
    int s0, int s1, int s2, int s3, int s4, int s5, int s6, int s7) {
    int t = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (t >= n_out || extent <= 0) return;
    int shp[8] = {s0, s1, s2, s3, s4, s5, s6, s7};
    int ost[8];
    int stride = 1;
    for (int d = 7; d >= 0; --d) {
        ost[d] = stride;
        stride *= shp[d];
    }
    int coords[8];
    int rem = t;
    for (int d = 7; d >= 0; --d) {
        int sd = (d == axis) ? 1 : shp[d];
        coords[d] = rem % sd;
        rem /= sd;
    }
    float acc = 0.f;
    for (int k = 0; k < extent; ++k) {
        coords[axis] = k;
        int idx = 0;
        for (int d = 0; d < 8; ++d) idx += coords[d] * ost[d];
        acc += in[idx];
    }
    out[t] = acc;
}

extern "C" __global__ void log_softmax_lastdim(const float* x, float* y, int rows, int k) {
    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= rows || k <= 0) return;
    const float* row = x + (long long)r * k;
    float* orow = y + (long long)r * k;
    float m = row[0];
    for (int j = 1; j < k; ++j) m = fmaxf(m, row[j]);
    float sum = 0.f;
    for (int j = 0; j < k; ++j) sum += expf(row[j] - m);
    float lse = m + logf(sum);
    for (int j = 0; j < k; ++j) orow[j] = row[j] - lse;
}

extern "C" __global__ void log_softmax_lastdim_bwd(const float* g, const float* y, float* dx,
                                                  int rows, int k) {
    int r = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (r >= rows || k <= 0) return;
    const float* grow = g + (long long)r * k;
    const float* yrow = y + (long long)r * k;
    float* drow = dx + (long long)r * k;
    float sg = 0.f;
    for (int j = 0; j < k; ++j) sg += grow[j];
    for (int j = 0; j < k; ++j) drow[j] = grow[j] - expf(yrow[j]) * sg;
}

/* 与 CPU AdamW.step 同一顺序，且每步都是 IEEE 舍入（禁止 FMA 收缩）。
   标量由主机按 numpy 2 的规则先收成 float32：例如 omb1 = float32(1 - b1)，
   不是 float32(1) - float32(b1)。wd_lr = float32(lr * wd)。原地写 p/m/v。 */
extern "C" __global__ void adamw_update(
    float* p, const float* g, float* m, float* v, int n,
    float b1, float omb1, float b2, float omb2,
    float lr, float wd_lr, float eps, float bc1, float bc2, int do_wd) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    float gi = g[i];
    float pi = p[i];
    if (do_wd) pi = __fsub_rn(pi, __fmul_rn(wd_lr, pi));
    float mi = __fadd_rn(__fmul_rn(b1, m[i]), __fmul_rn(omb1, gi));
    float g2 = __fmul_rn(gi, gi);
    float vi = __fadd_rn(__fmul_rn(b2, v[i]), __fmul_rn(omb2, g2));
    m[i] = mi;
    v[i] = vi;
    float mh = __fdiv_rn(mi, bc1);
    float vh = __fdiv_rn(vi, bc2);
    float denom = __fadd_rn(sqrtf(vh), eps);
    float step = __fdiv_rn(__fmul_rn(lr, mh), denom);
    p[i] = __fsub_rn(pi, step);
}

/* Muon 的动量。先 decoupled weight decay，再 buf = mom*buf + g，
   Nesterov 时 u = g + mom*buf（用刚写完的 buf）。 */
extern "C" __global__ void muon_momentum(
    float* p, const float* g, float* buf, float* u, int n,
    float mom, float wd_lr, int do_wd, int nesterov) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n) return;
    if (do_wd) p[i] = __fsub_rn(p[i], __fmul_rn(wd_lr, p[i]));
    float b = __fadd_rn(__fmul_rn(mom, buf[i]), g[i]);
    buf[i] = b;
    if (nesterov) u[i] = __fadd_rn(g[i], __fmul_rn(mom, b));
    else u[i] = b;
}

extern "C" __global__ void ewise_div_scalar(const float* a, float s, float* c, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) c[i] = __fdiv_rn(a[i], s);
}

extern "C" __global__ void ewise_sub_scale(float* p, const float* o, float scale, int n) {
    int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i < n) p[i] = __fsub_rn(p[i], __fmul_rn(scale, o[i]));
}

/* 行优先 C = A@B，多种 K 维累加顺序。用来对齐 numpy 的 matmul，mode 选定后训练走那一种。 */
extern "C" __global__ void sgemm_mode(
    const float* A, const float* B, float* C,
    int M, int N, int K, int mode) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = M * N;
    if (idx >= total || M <= 0 || N <= 0) return;
    int i = idx / N;
    int j = idx - i * N;
    const float* a = A + (long long)i * K;
    float acc = 0.f;
    if (K <= 0) {
        C[idx] = 0.f;
        return;
    }
    if (mode == 0) {
        for (int k = 0; k < K; ++k)
            acc = __fadd_rn(acc, __fmul_rn(a[k], B[(long long)k * N + j]));
    } else if (mode == 1) {
        for (int k = 0; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 2) {
        for (int k = K - 1; k >= 0; --k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 3) {
        float p[32];
        int n = K;
        if (n > 32) n = 32;
        for (int k = 0; k < n; ++k)
            p[k] = __fmul_rn(a[k], B[(long long)k * N + j]);
        int len = n;
        while (len > 1) {
            int m = len >> 1;
            for (int t = 0; t < m; ++t)
                p[t] = __fadd_rn(p[2 * t], p[2 * t + 1]);
            if (len & 1) {
                p[m] = p[len - 1];
                m += 1;
            }
            len = m;
        }
        acc = p[0];
        for (int k = n; k < K; ++k)
            acc = __fadd_rn(acc, __fmul_rn(a[k], B[(long long)k * N + j]));
    } else if (mode == 4) {
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
            a2 = __fmaf_rn(a[k + 2], B[(long long)(k + 2) * N + j], a2);
            a3 = __fmaf_rn(a[k + 3], B[(long long)(k + 3) * N + j], a3);
        }
        acc = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 5) {
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            float s = 0.f;
            s = __fmaf_rn(a[k], B[(long long)k * N + j], s);
            s = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], s);
            s = __fmaf_rn(a[k + 2], B[(long long)(k + 2) * N + j], s);
            s = __fmaf_rn(a[k + 3], B[(long long)(k + 3) * N + j], s);
            acc = __fadd_rn(acc, s);
        }
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 6) {
        float a0 = 0.f, a1 = 0.f;
        int k = 0;
        for (; k + 2 <= K; k += 2) {
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
        }
        acc = __fadd_rn(a0, a1);
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 7) {
        for (int k = 0; k < K; ++k)
            acc = __fadd_rn(acc, __fmul_rn(B[(long long)k * N + j], a[k]));
    } else if (mode == 8) {
        /* 每 8 个 K 做一对累加器，再加进 acc；尾巴继续一对累加器。 */
        int k = 0;
        for (; k + 8 <= K; k += 8) {
            float a0 = 0.f, a1 = 0.f;
            for (int t = 0; t < 8; t += 2) {
                a0 = __fmaf_rn(a[k + t], B[(long long)(k + t) * N + j], a0);
                a1 = __fmaf_rn(a[k + t + 1], B[(long long)(k + t + 1) * N + j], a1);
            }
            acc = __fadd_rn(acc, __fadd_rn(a0, a1));
        }
        float b0 = 0.f, b1 = 0.f;
        int t = k;
        for (; t + 1 < K; t += 2) {
            b0 = __fmaf_rn(a[t], B[(long long)t * N + j], b0);
            b1 = __fmaf_rn(a[t + 1], B[(long long)(t + 1) * N + j], b1);
        }
        acc = __fadd_rn(acc, __fadd_rn(b0, b1));
        if (t < K) acc = __fmaf_rn(a[t], B[(long long)t * N + j], acc);
    } else if (mode == 9) {
        /* 前 8 个一对累加器，剩下的顺序 fma 加在和上。 */
        int n8 = K < 8 ? K : 8;
        if (n8 & 1) n8 -= 1;
        float a0 = 0.f, a1 = 0.f;
        int k = 0;
        for (; k < n8; k += 2) {
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
        }
        acc = __fadd_rn(a0, a1);
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 10) {
        /* 4 路交错累加器，最后 (a0+a1)+(a2+a3)。 */
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int k = 0;
        for (; k + 3 < K; k += 4) {
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
            a2 = __fmaf_rn(a[k + 2], B[(long long)(k + 2) * N + j], a2);
            a3 = __fmaf_rn(a[k + 3], B[(long long)(k + 3) * N + j], a3);
        }
        acc = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else if (mode == 11) {
        /* 前 8 个 4 路累加器，剩下顺序 fma。 */
        float a0 = 0.f, a1 = 0.f, a2 = 0.f, a3 = 0.f;
        int n = K < 8 ? K : 8;
        int k = 0;
        for (; k + 3 < n; k += 4) {
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
            a2 = __fmaf_rn(a[k + 2], B[(long long)(k + 2) * N + j], a2);
            a3 = __fmaf_rn(a[k + 3], B[(long long)(k + 3) * N + j], a3);
        }
        acc = __fadd_rn(__fadd_rn(a0, a1), __fadd_rn(a2, a3));
        for (; k < K; ++k)
            acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    } else {
        /* mode 12：整段 2 路，但 a0+a1 的加法换成 fma 形式的顺序尾巴从 0 开始分块 4。 */
        int k = 0;
        for (; k + 4 <= K; k += 4) {
            float a0 = 0.f, a1 = 0.f;
            a0 = __fmaf_rn(a[k], B[(long long)k * N + j], a0);
            a1 = __fmaf_rn(a[k + 1], B[(long long)(k + 1) * N + j], a1);
            a0 = __fmaf_rn(a[k + 2], B[(long long)(k + 2) * N + j], a0);
            a1 = __fmaf_rn(a[k + 3], B[(long long)(k + 3) * N + j], a1);
            acc = __fadd_rn(acc, __fadd_rn(a0, a1));
        }
        float b0 = 0.f, b1 = 0.f;
        int t = k;
        for (; t + 1 < K; t += 2) {
            b0 = __fmaf_rn(a[t], B[(long long)t * N + j], b0);
            b1 = __fmaf_rn(a[t + 1], B[(long long)(t + 1) * N + j], b1);
        }
        acc = __fadd_rn(acc, __fadd_rn(b0, b1));
        if (t < K) acc = __fmaf_rn(a[t], B[(long long)t * N + j], acc);
    }
    C[idx] = acc;
}

/* K 维按给定切块累加。kind=1 块内顺序 fma；kind=2 块内奇偶两路 fma 再相加。块与块左加。 */
extern "C" __global__ void sgemm_compose(
    const float* A, const float* B, float* C,
    int M, int N, int K, int nchunks, int kind,
    int c0, int c1, int c2, int c3, int c4, int c5, int c6, int c7) {
    int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    int total = M * N;
    if (idx >= total) return;
    int i = idx / N;
    int j = idx - i * N;
    const float* a = A + (long long)i * K;
    int chunks[8] = {c0, c1, c2, c3, c4, c5, c6, c7};
    float acc = 0.f;
    int k = 0;
    int nch = nchunks;
    if (nch > 8) nch = 8;
    for (int c = 0; c < nch; ++c) {
        int len = chunks[c];
        if (len <= 0) break;
        float part;
        if (kind == 2) {
            float a0 = 0.f, a1 = 0.f;
            int t = 0;
            for (; t + 1 < len; t += 2) {
                a0 = __fmaf_rn(a[k + t], B[(long long)(k + t) * N + j], a0);
                a1 = __fmaf_rn(a[k + t + 1], B[(long long)(k + t + 1) * N + j], a1);
            }
            part = __fadd_rn(a0, a1);
            if (t < len)
                part = __fmaf_rn(a[k + t], B[(long long)(k + t) * N + j], part);
        } else {
            part = 0.f;
            for (int t = 0; t < len; ++t)
                part = __fmaf_rn(a[k + t], B[(long long)(k + t) * N + j], part);
        }
        acc = __fadd_rn(acc, part);
        k += len;
    }
    for (; k < K; ++k)
        acc = __fmaf_rn(a[k], B[(long long)k * N + j], acc);
    C[idx] = acc;
}

/* 顺序累加平方和，一个线程。n 很小（优化器里的矩阵）。和 np.sum(x*x) 的左折叠对齐。 */
extern "C" __global__ void sumsq_sequential(const float* a, float* out, int n) {
    if (threadIdx.x != 0 || blockIdx.x != 0) return;
    float s = 0.f;
    for (int i = 0; i < n; ++i) {
        float v = a[i];
        s = __fadd_rn(s, __fmul_rn(v, v));
    }
    out[0] = s;
}
"""

# MoE 派发/收集。专家 GEMM 仍走已有的 cublas/sgemm_mode，这里只做索引。
# top-k：logit 更大者优先，平局保留更小的专家下标（与 moeblk 的 strict-greater argmax 一致）。
_MOE_KERNELS = r"""
extern "C" __global__ void moe_topk(const float* logits, int* idx, float* val,
                                     int N, int E, int K) {
    int t = (int)blockIdx.x;
    if (t >= N || threadIdx.x != 0) return;
    const float* row = logits + (long long)t * E;
    int* irow = idx + (long long)t * K;
    float* vrow = val + (long long)t * K;
    for (int s = 0; s < K; ++s) {
        int best_e = -1;
        float best_v = 0.f;
        for (int e = 0; e < E; ++e) {
            int used = 0;
            for (int p = 0; p < s; ++p) {
                if (irow[p] == e) used = 1;
            }
            if (used) continue;
            float v = row[e];
            if (best_e < 0 || v > best_v || (v == best_v && e < best_e)) {
                best_e = e;
                best_v = v;
            }
        }
        irow[s] = best_e;
        vrow[s] = best_v;
    }
}

extern "C" __global__ void moe_sigmoid(const float* x, float* y, int n) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= n) return;
    float neg = __fmul_rn(x[i], -1.0f);
    float e = exp32_bits(neg);
    y[i] = __fdiv_rn(1.0f, __fadd_rn(1.0f, e));
}

extern "C" __global__ void moe_gather_pairs(const float* src, const int* idx, float* dst,
                                             int N, int E, int K) {
    int t = (int)blockIdx.x;
    if (t >= N) return;
    for (int s = (int)threadIdx.x; s < K; s += (int)blockDim.x) {
        int e = idx[(long long)t * K + s];
        dst[(long long)t * K + s] = src[(long long)t * E + e];
    }
}

/* 单线程按 token、slot 的顺序加，和 CPU 的 np.add.at 同一方向。 */
extern "C" __global__ void moe_scatter_pairs(const float* g, const int* idx, float* dst,
                                              int N, int E, int K) {
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 0; i < N * E; ++i) dst[i] = 0.f;
    for (int t = 0; t < N; ++t) {
        for (int s = 0; s < K; ++s) {
            int e = idx[(long long)t * K + s];
            dst[(long long)t * E + e] += g[(long long)t * K + s];
        }
    }
}

extern "C" __global__ void moe_compact(const int* idx, int* offsets, int* cursor,
                                        int* tok, int* slot, int N, int E, int K) {
    if (threadIdx.x || blockIdx.x) return;
    for (int e = 0; e < E; ++e) offsets[e] = 0;
    for (int t = 0; t < N; ++t) {
        for (int s = 0; s < K; ++s) offsets[idx[(long long)t * K + s]]++;
    }
    int sum = 0;
    for (int e = 0; e < E; ++e) {
        int c = offsets[e];
        offsets[e] = sum;
        cursor[e] = sum;
        sum += c;
    }
    offsets[E] = sum;
    for (int t = 0; t < N; ++t) {
        for (int s = 0; s < K; ++s) {
            int e = idx[(long long)t * K + s];
            int p = cursor[e]++;
            tok[p] = t;
            slot[p] = s;
        }
    }
}

extern "C" __global__ void moe_gather_rows(const float* x, const int* tok, float* out,
                                            int begin, int m, int d) {
    int i = (int)blockIdx.x;
    if (i >= m) return;
    int t = tok[begin + i];
    const float* src = x + (long long)t * d;
    float* dst = out + (long long)i * d;
    for (int j = (int)threadIdx.x; j < d; j += (int)blockDim.x) dst[j] = src[j];
}

extern "C" __global__ void moe_scatter_rows(const float* y, const int* tok, float* out,
                                             int begin, int m, int d) {
    int i = (int)blockIdx.x;
    if (i >= m) return;
    int t = tok[begin + i];
    const float* src = y + (long long)i * d;
    float* dst = out + (long long)t * d;
    for (int j = (int)threadIdx.x; j < d; j += (int)blockDim.x)
        atomicAdd(dst + j, src[j]);
}

extern "C" __global__ void moe_gather_slots(const float* w, const int* tok, const int* slot,
                                             float* out, int begin, int m, int K) {
    int i = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= m) return;
    int t = tok[begin + i];
    int s = slot[begin + i];
    out[i] = w[(long long)t * K + s];
}

extern "C" __global__ void moe_scatter_slots(const float* g, const int* tok, const int* slot,
                                              float* dst, int begin, int m, int N, int K) {
    if (threadIdx.x || blockIdx.x) return;
    for (int i = 0; i < N * K; ++i) dst[i] = 0.f;
    for (int i = 0; i < m; ++i) {
        int t = tok[begin + i];
        int s = slot[begin + i];
        dst[(long long)t * K + s] += g[i];
    }
}

extern "C" __global__ void moe_row_scale(const float* y, const float* w, float* out,
                                          int m, int d) {
    int i = (int)blockIdx.x;
    if (i >= m) return;
    float s = w[i];
    const float* src = y + (long long)i * d;
    float* dst = out + (long long)i * d;
    for (int j = (int)threadIdx.x; j < d; j += (int)blockDim.x)
        dst[j] = __fmul_rn(src[j], s);
}

extern "C" __global__ void moe_row_dot(const float* g, const float* y, float* dw,
                                        int m, int d) {
    int i = (int)blockIdx.x;
    if (i >= m || threadIdx.x != 0) return;
    const float* gg = g + (long long)i * d;
    const float* yy = y + (long long)i * d;
    float s = 0.f;
    for (int j = 0; j < d; ++j) s = __fadd_rn(s, __fmul_rn(gg[j], yy[j]));
    dw[i] = s;
}
"""


def _device_source():
    return (_CE_AND_HELPERS + "\n" + _M2_KERNELS + "\n" + _M3_KERNELS + "\n"
            + _M5_KERNELS + "\n" + _MOE_KERNELS)


class CudaError(RuntimeError):
    pass


class GpuBuf:
    """一块设备内存。由创建它的 CudaRuntime 释放。不要复制 ptr 到第二个 GpuBuf。"""

    __slots__ = ("rt", "ptr", "nbytes", "shape", "n", "dtype",
                 "known_scalar", "host_hold", "keepalive")

    def __init__(self, rt, ptr, nbytes, shape, n, dtype):
        self.rt = rt
        self.ptr = int(ptr)
        self.nbytes = int(nbytes)
        self.shape = tuple(shape)
        self.n = int(n)
        self.dtype = np.dtype(dtype)
        # 主机已经知道的标量（例如 backward() 填的 1）。读它不必再同步设备。
        self.known_scalar = None
        # 异步 H2D 期间主机缓冲必须活着。
        self.host_hold = None
        # 让还没跑完的 kernel 依赖的临时缓冲跟着本缓冲一起活。
        self.keepalive = None

    def __del__(self):
        ptr = getattr(self, "ptr", 0)
        rt = getattr(self, "rt", None)
        if ptr and rt is not None:
            self.ptr = 0
            try:
                rt.free(ptr)
            except Exception:
                pass

    def to_numpy(self):
        return self.rt.d2h(self)


def _dll_dirs():
    dirs = []
    extra = os.environ.get("TGAI_CUDA_BIN")
    if extra:
        dirs.append(extra)
    for key in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(key)
        if root:
            dirs.append(os.path.join(root, "bin"))
    toolkit = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if os.path.isdir(toolkit):
        try:
            names = sorted(os.listdir(toolkit), reverse=True)
        except OSError:
            names = []
        for name in names:
            dirs.append(os.path.join(toolkit, name, "bin"))
    # 不 import torch；只是如果旁边有 CUDA 版 torch 的 DLL 目录，允许拿来当搜索路径。
    for entry in list(sys.path):
        if not entry:
            continue
        cand = os.path.join(entry, "torch", "lib")
        if os.path.isdir(cand):
            dirs.append(cand)
    out = []
    seen = set()
    for d in dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        out.append(d)
    return out


def _prepare_dir(path):
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(path)
        except OSError:
            pass


def _find_file(names, dirs):
    for d in dirs:
        _prepare_dir(d)
        for name in names:
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    for name in names:
        for d in os.environ.get("PATH", "").split(os.pathsep):
            p = os.path.join(d, name)
            if os.path.isfile(p):
                return p
    return None


def _find_glob(pred, dirs):
    found = []
    for d in dirs:
        _prepare_dir(d)
        try:
            names = os.listdir(d)
        except OSError:
            continue
        for name in names:
            if pred(name) and os.path.isfile(os.path.join(d, name)):
                found.append(os.path.join(d, name))
    found.sort()
    return found[-1] if found else None


def _is_nvrtc_main(name):
    low = name.lower()
    return low.startswith("nvrtc64_") and low.endswith(".dll") and "builtin" not in low


def _is_cublas_main(name):
    low = name.lower()
    # 要 cublas64_12.dll，不要 cublasLt。
    return low.startswith("cublas64_") and low.endswith(".dll") and "lt" not in low


def _dedupe_dirs(dirs):
    out = []
    seen = set()
    for d in dirs:
        if not d or d in seen or not os.path.isdir(d):
            continue
        seen.add(d)
        out.append(d)
    return out


def _linux_lib_dirs():
    """Linux 上找 libcuda / libcublas / libnvrtc 的目录。不写 LD_LIBRARY_PATH。"""
    dirs = []
    extra = os.environ.get("TGAI_CUDA_BIN")
    if extra:
        dirs.append(extra)
    for key in ("CUDA_PATH", "CUDA_HOME"):
        root = os.environ.get(key)
        if root:
            dirs.append(os.path.join(root, "lib64"))
            dirs.append(os.path.join(root, "lib"))
    try:
        import sysconfig
        pure = sysconfig.get_path("purelib")
    except Exception:
        pure = None
    if pure:
        nvidia = os.path.join(pure, "nvidia")
        if os.path.isdir(nvidia):
            try:
                pkgs = sorted(os.listdir(nvidia))
            except OSError:
                pkgs = []
            for pkg in pkgs:
                lib = os.path.join(nvidia, pkg, "lib")
                if os.path.isdir(lib):
                    dirs.append(lib)
    dirs.append("/usr/lib/x86_64-linux-gnu")
    dirs.append("/usr/lib64")
    return _dedupe_dirs(dirs)


def _preload(path):
    if path and os.path.isfile(path):
        ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)


def _alias_driver_symbol(cu, name):
    """优先 _v2。Windows 上不带后缀的是旧 32 位 ABI，缺 _v2 就报错。

    Linux 驱动 610.57 上实测：cuMemAlloc 与 cuMemAlloc_v2 地址不同，
    不带后缀的 cuMemAlloc 在 primary context 上返回 201。所以这里也优先 _v2。
    只有 Linux 且库里根本没有 _v2 时，才退回不带后缀的符号。
    """
    v2 = getattr(cu, name + "_v2", None)
    if v2 is not None:
        setattr(cu, name, v2)
        return
    if sys.platform != "win32" and getattr(cu, name, None) is not None:
        return
    raise CudaError(
        "驱动库没有 %s_v2。不能用旧入口：primary context 上会 rc=201。" % name
    )


class CudaRuntime:
    def __init__(self):
        self.ready = False
        self.cu = None
        self.nvrtc = None
        self.cublas = None
        self.dev = None
        self.ctx = None
        self.handle = None
        self.mod = None
        self._fns = {}
        self._dirs = []
        self._h2d_cache = {}

    def ensure(self):
        if self.cu is None:
            self._init_driver()
        self._activate()
        if self.handle is None:
            self._init_cublas()
        if self.mod is None:
            self._init_kernels()
        self.ready = True
        return self

    def _init_driver(self):
        if sys.platform == "win32":
            self._dirs = _dll_dirs()
            for d in self._dirs:
                _prepare_dir(d)
            try:
                self.cu = ctypes.WinDLL("nvcuda.dll")
            except OSError as ex:
                raise CudaError(
                    "找不到 nvcuda.dll（NVIDIA 驱动没加载）。"
                    "当前进程建不了 CUDA 上下文。需要一台已装驱动的 NVIDIA GPU。"
                    " 原始错误: %s" % (ex,)
                ) from ex
        else:
            self._dirs = _linux_lib_dirs()
            path = _find_file(["libcuda.so.1", "libcuda.so"], self._dirs)
            try:
                if path:
                    self.cu = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
                else:
                    self.cu = ctypes.CDLL("libcuda.so.1", mode=ctypes.RTLD_GLOBAL)
            except OSError as ex:
                raise CudaError(
                    "找不到 libcuda.so.1（NVIDIA 驱动没加载）。已查目录: %s。"
                    " 原始错误: %s" % (self._dirs or ["<空>"], ex)
                ) from ex
        self._bind_driver()
        self._chk(self.cu.cuInit(0), "cuInit")
        dev = ctypes.c_int()
        rc = self.cu.cuDeviceGet(ctypes.byref(dev), 0)
        if rc != 0:
            raise CudaError("cuDeviceGet(0) 失败 rc=%d。机器上没有可用的 CUDA 设备。" % rc)
        self.dev = int(dev.value)
        ctx = ctypes.c_void_p()
        self._chk(self.cu.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), self.dev),
                  "cuDevicePrimaryCtxRetain")
        self.ctx = ctx
        self._activate()

    def _bind_driver(self):
        cu = self.cu
        # Windows 的 nvcuda.dll 同时导出旧符号和 _v2。cuda.h 里的
        # #define cuMemAlloc cuMemAlloc_v2 对 ctypes 无效。旧的 cuMemAlloc /
        # cuMemFree / cuMemcpy* 仍是 32 位 CUdeviceptr ABI，拿到
        # cuDevicePrimaryCtxRetain 的 context 会直接返回
        # CUDA_ERROR_INVALID_CONTEXT (201)，尽管 cuCtxGetCurrent 对得上。
        for name in ("cuMemAlloc", "cuMemFree", "cuMemcpyHtoD", "cuMemcpyDtoH", "cuMemcpyDtoD",
                     "cuMemcpyHtoDAsync", "cuMemcpyDtoHAsync", "cuMemsetD32Async"):
            _alias_driver_symbol(cu, name)
        cu.cuInit.argtypes = [ctypes.c_uint]
        cu.cuInit.restype = ctypes.c_int
        cu.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
        cu.cuDeviceGet.restype = ctypes.c_int
        cu.cuDeviceGetAttribute.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int, ctypes.c_int]
        cu.cuDeviceGetAttribute.restype = ctypes.c_int
        cu.cuDevicePrimaryCtxRetain.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_int]
        cu.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        cu.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        cu.cuCtxSetCurrent.restype = ctypes.c_int
        cu.cuCtxSynchronize.argtypes = []
        cu.cuCtxSynchronize.restype = ctypes.c_int
        cu.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.c_size_t]
        cu.cuMemAlloc.restype = ctypes.c_int
        cu.cuMemFree.argtypes = [ctypes.c_uint64]
        cu.cuMemFree.restype = ctypes.c_int
        cu.cuMemcpyHtoD.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t]
        cu.cuMemcpyHtoD.restype = ctypes.c_int
        cu.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuMemcpyDtoH.restype = ctypes.c_int
        cu.cuMemcpyDtoD.argtypes = [ctypes.c_uint64, ctypes.c_uint64, ctypes.c_size_t]
        cu.cuMemcpyDtoD.restype = ctypes.c_int
        cu.cuMemcpyHtoDAsync.argtypes = [ctypes.c_uint64, ctypes.c_void_p, ctypes.c_size_t, ctypes.c_void_p]
        cu.cuMemcpyHtoDAsync.restype = ctypes.c_int
        cu.cuMemcpyDtoHAsync.argtypes = [ctypes.c_void_p, ctypes.c_uint64, ctypes.c_size_t, ctypes.c_void_p]
        cu.cuMemcpyDtoHAsync.restype = ctypes.c_int
        cu.cuMemsetD32Async.argtypes = [ctypes.c_uint64, ctypes.c_uint, ctypes.c_size_t, ctypes.c_void_p]
        cu.cuMemsetD32Async.restype = ctypes.c_int
        cu.cuStreamSynchronize.argtypes = [ctypes.c_void_p]
        cu.cuStreamSynchronize.restype = ctypes.c_int
        cu.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p]
        cu.cuModuleLoadData.restype = ctypes.c_int
        cu.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
        cu.cuModuleGetFunction.restype = ctypes.c_int
        cu.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
            ctypes.c_uint, ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
        ]
        cu.cuLaunchKernel.restype = ctypes.c_int
        try:
            cu.cuGetErrorString.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
            cu.cuGetErrorString.restype = ctypes.c_int
        except AttributeError:
            pass

    def _activate(self):
        if self.ctx is not None:
            self._chk(self.cu.cuCtxSetCurrent(self.ctx), "cuCtxSetCurrent")

    def _chk(self, rc, what):
        if rc == 0:
            return
        extra = ""
        get = getattr(self.cu, "cuGetErrorString", None)
        if get is not None:
            msg = ctypes.c_char_p()
            if get(int(rc), ctypes.byref(msg)) == 0 and msg.value:
                extra = " " + msg.value.decode("utf-8", "replace")
        raise CudaError("%s 失败 rc=%d%s" % (what, int(rc), extra))

    def _cublas_path(self):
        if sys.platform == "win32":
            path = _find_glob(_is_cublas_main, self._dirs)
            if path is None:
                path = _find_file(["cublas64_12.dll", "cublas64_11.dll"], self._dirs)
            if path is None:
                raise CudaError(
                    "找不到 cublas64_*.dll。已查目录: %s" % (self._dirs or ["<空>"])
                )
            return path
        path = _find_file(["libcublas.so.12", "libcublas.so"], self._dirs)
        if path is None:
            raise CudaError(
                "找不到 libcublas.so.12。已查目录: %s" % (self._dirs or ["<空>"])
            )
        return path

    def _init_cublas(self):
        path = self._cublas_path()
        folder = os.path.dirname(path)
        _prepare_dir(folder)
        if sys.platform != "win32":
            # 同目录预加载 cublasLt。这台机器的 libcublas 自带 $ORIGIN，
            # 直接 CDLL 也能开；预加载是为了不依赖 LD_LIBRARY_PATH。
            for name in ("libcublasLt.so.12", "libcublasLt.so"):
                try:
                    _preload(os.path.join(folder, name))
                except OSError:
                    pass
        try:
            if sys.platform == "win32":
                self.cublas = ctypes.CDLL(path)
            else:
                self.cublas = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as ex:
            raise CudaError("加载 %s 失败: %s" % (path, ex)) from ex
        self._cublas_path = path
        cb = self.cublas
        cb.cublasCreate_v2.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        cb.cublasCreate_v2.restype = ctypes.c_int
        cb.cublasSetMathMode.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cb.cublasSetMathMode.restype = ctypes.c_int
        cb.cublasSetPointerMode_v2.argtypes = [ctypes.c_void_p, ctypes.c_int]
        cb.cublasSetPointerMode_v2.restype = ctypes.c_int
        cb.cublasSetStream_v2.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        cb.cublasSetStream_v2.restype = ctypes.c_int
        # 设备指针用 c_uint64。c_void_p 是有符号的，地址最高位为 1 时会 OverflowError。
        cb.cublasSgemm_v2.argtypes = [
            ctypes.c_void_p,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_int,
            ctypes.c_uint64, ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_uint64, ctypes.c_int,
        ]
        cb.cublasSgemm_v2.restype = ctypes.c_int
        try:
            cb.cublasGetStatusString.argtypes = [ctypes.c_int]
            cb.cublasGetStatusString.restype = ctypes.c_char_p
        except AttributeError:
            pass
        handle = ctypes.c_void_p()
        self._chk_blas(cb.cublasCreate_v2(ctypes.byref(handle)), "cublasCreate_v2")
        self.handle = handle
        self._chk_blas(cb.cublasSetPointerMode_v2(handle, CUBLAS_POINTER_MODE_HOST),
                       "cublasSetPointerMode_v2")
        self._chk_blas(cb.cublasSetMathMode(handle, CUBLAS_PEDANTIC_MATH),
                       "cublasSetMathMode(PEDANTIC)")
        self._chk_blas(cb.cublasSetStream_v2(handle, ctypes.c_void_p(0)),
                       "cublasSetStream_v2")

    def _chk_blas(self, rc, what):
        if rc == 0:
            return
        extra = ""
        get = getattr(self.cublas, "cublasGetStatusString", None)
        if get is not None:
            try:
                s = get(int(rc))
                if s:
                    extra = " " + s.decode("utf-8", "replace")
            except Exception:
                pass
        raise CudaError("%s 失败 rc=%d%s" % (what, int(rc), extra))

    def _cc_arch(self):
        major = ctypes.c_int()
        minor = ctypes.c_int()
        self._chk(self.cu.cuDeviceGetAttribute(
            ctypes.byref(major), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, self.dev),
            "compute capability major")
        self._chk(self.cu.cuDeviceGetAttribute(
            ctypes.byref(minor), CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, self.dev),
            "compute capability minor")
        return "compute_%d%d" % (int(major.value), int(minor.value))

    def _nvrtc_path(self):
        if sys.platform == "win32":
            path = _find_glob(_is_nvrtc_main, self._dirs)
            if path is None:
                raise CudaError(
                    "找不到 nvrtc64_*.dll（NVRTC）。已查目录: %s" % (self._dirs or ["<空>"])
                )
            return path
        path = _find_file(["libnvrtc.so.12", "libnvrtc.so"], self._dirs)
        if path is None:
            raise CudaError(
                "找不到 libnvrtc.so.12（NVRTC）。已查目录: %s" % (self._dirs or ["<空>"])
            )
        return path

    def _init_kernels(self):
        path = self._nvrtc_path()
        folder = os.path.dirname(path)
        _prepare_dir(folder)
        if sys.platform != "win32":
            try:
                names = os.listdir(folder)
            except OSError:
                names = []
            for name in names:
                if name.startswith("libnvrtc-builtins.so"):
                    try:
                        _preload(os.path.join(folder, name))
                    except OSError:
                        pass
        try:
            if sys.platform == "win32":
                self.nvrtc = ctypes.WinDLL(path)
            else:
                self.nvrtc = ctypes.CDLL(path, mode=ctypes.RTLD_GLOBAL)
        except OSError as ex:
            raise CudaError("加载 %s 失败: %s" % (path, ex)) from ex
        self._bind_nvrtc()
        arch = self._cc_arch()
        ptx = self._compile(arch)
        mod = ctypes.c_void_p()
        buf = ctypes.create_string_buffer(ptx)
        self._chk(self.cu.cuModuleLoadData(
            ctypes.byref(mod), ctypes.cast(buf, ctypes.c_void_p)), "cuModuleLoadData")
        # PTX 缓冲必须活过 load；cuModuleLoadData 会拷贝，但保留引用更稳。
        self._ptx_keep = buf
        self.mod = mod

    def _bind_nvrtc(self):
        n = self.nvrtc
        n.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        n.nvrtcCreateProgram.restype = ctypes.c_int
        n.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        n.nvrtcCompileProgram.restype = ctypes.c_int
        n.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        n.nvrtcGetPTXSize.restype = ctypes.c_int
        n.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        n.nvrtcGetPTX.restype = ctypes.c_int
        n.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        n.nvrtcGetProgramLogSize.restype = ctypes.c_int
        n.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        n.nvrtcGetProgramLog.restype = ctypes.c_int
        n.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        n.nvrtcDestroyProgram.restype = ctypes.c_int

    def _compile(self, arch):
        # 源码没有 #include。不要把带空格的 CUDA include 塞进 -I，NVRTC 会把空格再拆开。
        opts = [
            "--gpu-architecture=%s" % arch,
            "--std=c++14",
            "--device-as-default-execution-space",
        ]
        opts_b = [o.encode("utf-8") for o in opts]
        arr = (ctypes.c_char_p * len(opts_b))(*opts_b)
        prog = ctypes.c_void_p()
        rc = self.nvrtc.nvrtcCreateProgram(
            ctypes.byref(prog), _device_source().encode("utf-8"), b"tgai_train.cu", 0, None, None)
        self._chk_nvrtc(rc, "nvrtcCreateProgram", prog)
        rc = self.nvrtc.nvrtcCompileProgram(prog, len(opts_b), arr)
        log = self._nvrtc_log(prog)
        if rc != 0:
            self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
            raise CudaError("NVRTC 编译失败 arch=%s\n%s" % (arch, log))
        sz = ctypes.c_size_t()
        self._chk_nvrtc(self.nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(sz)), "nvrtcGetPTXSize", prog)
        buf = ctypes.create_string_buffer(sz.value)
        self._chk_nvrtc(self.nvrtc.nvrtcGetPTX(prog, buf), "nvrtcGetPTX", prog)
        self.nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))
        return buf.raw

    def _chk_nvrtc(self, rc, what, prog):
        if rc == 0:
            return
        log = self._nvrtc_log(prog) if prog else ""
        raise CudaError("%s 失败 rc=%d\n%s" % (what, int(rc), log))

    def _nvrtc_log(self, prog):
        if not prog:
            return ""
        sz = ctypes.c_size_t()
        if self.nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(sz)) != 0 or sz.value <= 1:
            return ""
        buf = ctypes.create_string_buffer(sz.value)
        self.nvrtc.nvrtcGetProgramLog(prog, buf)
        return buf.value.decode("utf-8", "replace")

    def sync(self):
        self._activate()
        self._chk(self.cu.cuCtxSynchronize(), "cuCtxSynchronize")

    def free(self, ptr):
        if not ptr or self.cu is None:
            return
        try:
            if self.ctx is not None:
                self.cu.cuCtxSetCurrent(self.ctx)
                self.cu.cuCtxSynchronize()
            self.cu.cuMemFree(ctypes.c_uint64(int(ptr)))
        except Exception:
            pass

    def _alloc(self, nbytes):
        self._activate()
        nbytes = int(nbytes)
        if nbytes <= 0:
            raise CudaError("cuMemAlloc 拒绝 0 字节")
        ptr = ctypes.c_uint64()
        self._chk(self.cu.cuMemAlloc(ctypes.byref(ptr), ctypes.c_size_t(nbytes)), "cuMemAlloc")
        if not ptr.value:
            raise CudaError("cuMemAlloc 返回空指针")
        return int(ptr.value)

    def empty(self, shape, dtype=np.float32):
        shape = tuple(shape)
        dt = np.dtype(dtype)
        n = 1
        for s in shape:
            n *= int(s)
        # cuMemAlloc(0) 会返回 INVALID_VALUE，空张量也留 1 个元素的垫片，n 仍记 0。
        nbytes = max(n, 1) * int(dt.itemsize)
        ptr = self._alloc(nbytes)
        return GpuBuf(self, ptr, nbytes, shape, n, dt)

    def _memset32(self, buf, word):
        """按 32 位字填充。走默认流的异步 memset，不把前面的 kernel 同步回来。"""
        n = int(buf.n)
        if n <= 0:
            return
        self._activate()
        self._chk(self.cu.cuMemsetD32Async(
            ctypes.c_uint64(buf.ptr),
            ctypes.c_uint(int(word) & 0xFFFFFFFF),
            ctypes.c_size_t(n),
            ctypes.c_void_p(0)), "cuMemsetD32Async")

    def h2d(self, arr, dtype=np.float32, cache_slot=None):
        """异步 H2D。cache_slot 命中时复用上一份设备缓冲，避免每步重传小索引。"""
        src = np.ascontiguousarray(np.asarray(arr), dtype=dtype)
        if cache_slot:
            key = (src.shape, src.dtype.str, src.tobytes())
            hit = self._h2d_cache.get(cache_slot)
            if hit is not None and hit[0] == key:
                return hit[1]
        buf = self.empty(src.shape, dtype=dtype)
        if src.nbytes:
            # 调用方的数组可能马上被改掉，异步拷贝必须抓住自己的一份。
            hold = np.array(src, copy=True, order="C")
            self._activate()
            self._chk(self.cu.cuMemcpyHtoDAsync(
                ctypes.c_uint64(buf.ptr),
                hold.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(int(hold.nbytes)),
                ctypes.c_void_p(0)), "cuMemcpyHtoDAsync")
            buf.host_hold = hold
        if buf.n == 1 and np.issubdtype(buf.dtype, np.floating):
            buf.known_scalar = float(src.reshape(-1)[0])
        if cache_slot:
            self._h2d_cache[cache_slot] = ((src.shape, src.dtype.str, src.tobytes()), buf)
        return buf

    def d2h(self, buf):
        """只同步默认流。拷贝排在这条流已有工作之后，不再先 cuCtxSynchronize 整颗设备。"""
        self._activate()
        out = np.empty(buf.shape, dtype=buf.dtype)
        if buf.n:
            self._chk(self.cu.cuMemcpyDtoHAsync(
                out.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_uint64(buf.ptr),
                ctypes.c_size_t(int(out.nbytes)),
                ctypes.c_void_p(0)), "cuMemcpyDtoHAsync")
            self._chk(self.cu.cuStreamSynchronize(ctypes.c_void_p(0)), "cuStreamSynchronize")
        return out

    def clone(self, buf):
        out = self.empty(buf.shape, dtype=buf.dtype)
        if buf.n:
            self._chk(self.cu.cuMemcpyDtoD(
                ctypes.c_uint64(out.ptr),
                ctypes.c_uint64(buf.ptr),
                ctypes.c_size_t(int(buf.n) * int(buf.dtype.itemsize))), "cuMemcpyDtoD")
        out.known_scalar = buf.known_scalar
        return out

    def full(self, shape, value, dtype=np.float32):
        dt = np.dtype(dtype)
        if dt.itemsize != 4:
            return self.h2d(np.full(tuple(shape), value, dtype=dt), dtype=dt)
        buf = self.empty(shape, dtype=dt)
        if buf.n:
            word = int(np.array([value], dtype=dt).view(np.uint32)[0])
            self._memset32(buf, word)
        if buf.n == 1 and np.issubdtype(dt, np.floating):
            buf.known_scalar = float(np.asarray(value, dtype=np.float32))
        return buf

    def read_scalar(self, buf):
        if int(buf.n) != 1:
            raise CudaError("期望标量梯度，收到 shape=%s" % (buf.shape,))
        if buf.known_scalar is not None:
            return float(buf.known_scalar)
        return float(np.asarray(self.d2h(buf), dtype=np.float64).reshape(-1)[0])

    def _fn(self, name):
        fn = self._fns.get(name)
        if fn is not None:
            return fn
        out = ctypes.c_void_p()
        self._chk(self.cu.cuModuleGetFunction(ctypes.byref(out), self.mod, name.encode("ascii")),
                  "cuModuleGetFunction(%s)" % name)
        self._fns[name] = out
        return out

    def launch(self, name, grid, block, args, shared=0):
        self._activate()
        fn = self._fn(name)
        holders = []
        ptrs = (ctypes.c_void_p * len(args))()
        for i, (val, typ) in enumerate(args):
            obj = val if isinstance(val, ctypes._SimpleCData) else typ(val)
            holders.append(obj)
            ptrs[i] = ctypes.cast(ctypes.byref(obj), ctypes.c_void_p)
        rc = self.cu.cuLaunchKernel(
            fn,
            int(grid[0]), int(grid[1]), int(grid[2]),
            int(block[0]), int(block[1]), int(block[2]),
            int(shared), ctypes.c_void_p(0), ptrs, None)
        self._chk(rc, "cuLaunchKernel(%s)" % name)
        return holders

    def _ewise_grid(self, n):
        block = 256
        grid = (int(n) + block - 1) // block
        if grid < 1:
            grid = 1
        return (grid, 1, 1), (block, 1, 1)

    def _ptr(self, buf):
        return (int(buf.ptr), ctypes.c_uint64)

    def transpose2d(self, buf):
        if len(buf.shape) != 2:
            raise CudaError("transpose2d 只支持 2D，收到 %s" % (buf.shape,))
        rows, cols = int(buf.shape[0]), int(buf.shape[1])
        out = self.empty((cols, rows), dtype=np.float32)
        n = rows * cols
        if n:
            grid, block = self._ewise_grid(n)
            self.launch("transpose2d", grid, block, [
                self._ptr(buf), self._ptr(out),
                (rows, ctypes.c_int), (cols, ctypes.c_int),
            ])
        return out

    def add(self, a, b):
        if tuple(a.shape) == tuple(b.shape):
            out = self.empty(a.shape, dtype=np.float32)
            if a.n:
                grid, block = self._ewise_grid(a.n)
                self.launch("ewise_add", grid, block, [
                    self._ptr(a), self._ptr(b), self._ptr(out),
                    (int(a.n), ctypes.c_int),
                ])
            return out
        return self.add_broadcast(a, b)

    def _broadcast_shapes(self, sa, sb):
        sa = tuple(int(s) for s in sa)
        sb = tuple(int(s) for s in sb)
        ndim = max(len(sa), len(sb), 1)
        if ndim > 4:
            raise CudaError("broadcast 最高 4 维，收到 %s 和 %s" % (sa, sb))
        pa = (1,) * (ndim - len(sa)) + sa
        pb = (1,) * (ndim - len(sb)) + sb
        out = []
        for x, y in zip(pa, pb):
            if x == y:
                out.append(x)
            elif x == 1:
                out.append(y)
            elif y == 1:
                out.append(x)
            else:
                raise CudaError("形状不能广播: %s vs %s" % (sa, sb))
        return tuple(out), pa, pb

    def _pad4(self, shape):
        shape = tuple(int(s) for s in shape)
        if len(shape) > 4:
            raise CudaError("内部 rank %d > 4" % len(shape))
        return (1,) * (4 - len(shape)) + shape

    def _strides4(self, padded):
        """长度为 1 的维步长记 0，这样广播维不会读出源缓冲。"""
        st = [0, 0, 0, 0]
        stride = 1
        for i in range(3, -1, -1):
            st[i] = 0 if padded[i] == 1 else stride
            stride *= int(padded[i])
        return st

    def add_broadcast(self, a, b):
        out_shape, pa, pb = self._broadcast_shapes(a.shape, b.shape)
        out = self.empty(out_shape, dtype=np.float32)
        n = int(out.n)
        if n == 0:
            return out
        if n > 2147483647:
            raise CudaError("broadcast add 元素个数超过 int32")
        c = self._pad4(out_shape)
        sa = self._strides4(self._pad4(pa))
        sb = self._strides4(self._pad4(pb))
        grid, block = self._ewise_grid(n)
        self.launch("bcast_add4", grid, block, [
            self._ptr(a), self._ptr(b), self._ptr(out),
            (n, ctypes.c_int),
            (c[0], ctypes.c_int), (c[1], ctypes.c_int),
            (c[2], ctypes.c_int), (c[3], ctypes.c_int),
            (sa[0], ctypes.c_int), (sa[1], ctypes.c_int),
            (sa[2], ctypes.c_int), (sa[3], ctypes.c_int),
            (sb[0], ctypes.c_int), (sb[1], ctypes.c_int),
            (sb[2], ctypes.c_int), (sb[3], ctypes.c_int),
        ])
        return out

    def _left_pad8(self, values, fill):
        values = tuple(int(v) for v in values)
        if len(values) > 8:
            raise CudaError("rank %d > 8" % len(values))
        return (int(fill),) * (8 - len(values)) + values

    def _row_strides(self, shape):
        shape = tuple(int(s) for s in shape)
        raw = [0] * len(shape)
        stride = 1
        for i in range(len(shape) - 1, -1, -1):
            raw[i] = stride
            stride *= shape[i]
        return raw

    def sum_axis(self, buf, axis, keepdims=False):
        """沿一轴求和。keepdims=False 时丢掉这一维。值留在 GPU。"""
        shape = tuple(int(s) for s in buf.shape)
        nd = len(shape)
        if nd == 0:
            raise CudaError("sum_axis 不能用于标量")
        if nd > 8:
            raise CudaError("sum_axis 最高 8 维，收到 %s" % (shape,))
        axis = int(axis)
        if axis < 0:
            axis += nd
        if axis < 0 or axis >= nd:
            raise CudaError("sum_axis 轴 %d 越界（ndim=%d）" % (axis, nd))
        extent = int(shape[axis])
        out_keep = list(shape)
        out_keep[axis] = 1
        if extent == 0:
            out = self.zeros(tuple(out_keep), dtype=np.float32)
        else:
            out = self.empty(tuple(out_keep), dtype=np.float32)
            n_out = int(out.n)
            if n_out:
                shp8 = self._left_pad8(shape, 1)
                axis8 = axis + (8 - nd)
                grid, block = self._ewise_grid(n_out)
                self.launch("sum_axis8", grid, block, [
                    self._ptr(buf), self._ptr(out),
                    (n_out, ctypes.c_int), (axis8, ctypes.c_int), (extent, ctypes.c_int),
                ] + [(int(s), ctypes.c_int) for s in shp8])
        if keepdims:
            return out
        dropped = shape[:axis] + shape[axis + 1:]
        return self.reshape(out, dropped)

    def unbroadcast(self, g, shape):
        """把广播后的梯度加回原形状。规则与主机 _unbroadcast 相同，求和在 GPU 上做。"""
        shape = tuple(int(s) for s in shape)
        cur = g
        if tuple(int(s) for s in cur.shape) == shape:
            return cur
        while len(cur.shape) > len(shape):
            cur = self.sum_axis(cur, 0, keepdims=False)
        for axis, n in enumerate(shape):
            if n == 1 and int(cur.shape[axis]) != 1:
                cur = self.sum_axis(cur, axis, keepdims=True)
        if tuple(int(s) for s in cur.shape) != shape:
            cur = self.reshape(cur, shape)
        return cur

    def broadcast_to(self, buf, shape):
        """右对齐广播到 shape。已经同形就返回原缓冲。"""
        shape = tuple(int(s) for s in shape)
        src = tuple(int(s) for s in buf.shape)
        if src == shape:
            return buf
        if len(shape) > 8 or len(src) > len(shape):
            raise CudaError("broadcast_to %s -> %s" % (src, shape))
        if any(s < 0 for s in shape):
            raise CudaError("broadcast_to 目标形状非法 %s" % (shape,))
        ndim = len(shape)
        aligned = (1,) * (ndim - len(src)) + src
        for a, b in zip(aligned, shape):
            if a != b and a != 1:
                raise CudaError("不能广播 %s -> %s" % (src, shape))
        raw = self._row_strides(src)
        strides = []
        base = ndim - len(src)
        for i, a in enumerate(aligned):
            if i < base or a == 1:
                strides.append(0)
            else:
                strides.append(int(raw[i - base]))
        out = self.empty(shape, dtype=np.float32)
        n = int(out.n)
        if n:
            if n > 2147483647:
                raise CudaError("broadcast_to 元素个数超过 int32")
            grid, block = self._ewise_grid(n)
            self.launch("broadcast_to8", grid, block, [
                self._ptr(buf), self._ptr(out), (n, ctypes.c_int),
            ] + [(int(s), ctypes.c_int) for s in self._left_pad8(shape, 1)]
              + [(int(s), ctypes.c_int) for s in self._left_pad8(strides, 0)])
        return out

    def permute(self, buf, axes):
        """按轴排列。输出是新的连续缓冲。axes 是输入轴的一个排列。"""
        axes = tuple(int(a) for a in axes)
        in_shape = tuple(int(s) for s in buf.shape)
        nd = len(in_shape)
        if nd > 8:
            raise CudaError("permute 最高 8 维，收到 %s" % (in_shape,))
        if sorted(axes) != list(range(nd)):
            raise CudaError("permute 不是排列: %s" % (axes,))
        out_shape = tuple(in_shape[a] for a in axes)
        in_stride = self._row_strides(in_shape)
        src_stride = [int(in_stride[a]) for a in axes]
        out = self.empty(out_shape, dtype=np.float32)
        n = int(out.n)
        if n:
            if n > 2147483647:
                raise CudaError("permute 元素个数超过 int32")
            grid, block = self._ewise_grid(n)
            self.launch("permute8", grid, block, [
                self._ptr(buf), self._ptr(out), (n, ctypes.c_int),
            ] + [(int(s), ctypes.c_int) for s in self._left_pad8(out_shape, 1)]
              + [(int(s), ctypes.c_int) for s in self._left_pad8(src_stride, 0)])
        return out

    def _broadcast_lead(self, sa, sb):
        sa = tuple(int(s) for s in sa)
        sb = tuple(int(s) for s in sb)
        ndim = max(len(sa), len(sb))
        if ndim == 0:
            return ()
        if ndim > 6:
            raise CudaError("matmul 前导维 %d 超过 6" % ndim)
        pa = (1,) * (ndim - len(sa)) + sa
        pb = (1,) * (ndim - len(sb)) + sb
        out = []
        for x, y in zip(pa, pb):
            if x == y:
                out.append(x)
            elif x == 1:
                out.append(y)
            elif y == 1:
                out.append(x)
            else:
                raise CudaError("matmul 前导维不能广播: %s vs %s" % (sa, sb))
        return tuple(out)

    def matmul_general(self, A, B):
        """最后两维做矩阵乘，前导维按 PyTorch 右对齐广播。没有前导维时就是 sgemm。"""
        if len(A.shape) < 2 or len(B.shape) < 2:
            raise CudaError("matmul 至少 2D，收到 %s @ %s" % (A.shape, B.shape))
        if len(A.shape) > 8 or len(B.shape) > 8:
            raise CudaError("matmul 最高 8 维，收到 %s @ %s" % (A.shape, B.shape))
        M, K = int(A.shape[-2]), int(A.shape[-1])
        K2, N = int(B.shape[-2]), int(B.shape[-1])
        if K != K2:
            raise CudaError("matmul K 不匹配: %s @ %s" % (A.shape, B.shape))
        a_lead = tuple(int(s) for s in A.shape[:-2])
        b_lead = tuple(int(s) for s in B.shape[:-2])
        if not a_lead and not b_lead:
            return self.sgemm(A, B)
        out_lead = self._broadcast_lead(a_lead, b_lead)
        A2 = self.broadcast_to(A, out_lead + (M, K))
        B2 = self.broadcast_to(B, out_lead + (K, N))
        batch = 1
        for s in out_lead:
            batch *= int(s)
        C3 = self.bmm(
            self.reshape(A2, (batch, M, K)),
            self.reshape(B2, (batch, K, N)),
        )
        return self.reshape(C3, out_lead + (M, N))

    def reshape(self, buf, shape):
        """C 顺序重解释。字节序不变，所以反向也只是再解释一次。"""
        shape = tuple(int(s) for s in shape)
        if any(s < 0 for s in shape):
            raise CudaError("reshape 不支持负数维度 %s" % (shape,))
        n = 1
        for s in shape:
            n *= int(s)
        if n != int(buf.n):
            raise CudaError("reshape %s -> %s 元素个数 %d != %d" % (buf.shape, shape, buf.n, n))
        out = self.empty(shape, dtype=buf.dtype)
        if n:
            self._activate()
            self._chk(self.cu.cuMemcpyDtoD(
                ctypes.c_uint64(out.ptr),
                ctypes.c_uint64(buf.ptr),
                ctypes.c_size_t(n * int(buf.dtype.itemsize))), "cuMemcpyDtoD(reshape)")
        return out

    def swap_last2(self, buf):
        if len(buf.shape) < 2:
            raise CudaError("swap_last2 至少要 2 维，收到 %s" % (buf.shape,))
        dims = tuple(int(s) for s in buf.shape)
        m, n = dims[-2], dims[-1]
        batch = 1
        for s in dims[:-2]:
            batch *= int(s)
        out = self.empty(dims[:-2] + (n, m), dtype=np.float32)
        total = batch * m * n
        if total and m > 0 and n > 0:
            grid, block = self._ewise_grid(total)
            self.launch("swap_last2", grid, block, [
                self._ptr(buf), self._ptr(out),
                (batch, ctypes.c_int), (m, ctypes.c_int), (n, ctypes.c_int),
            ])
        return out

    def bmm(self, A, B):
        """行优先 C[b,M,N] = A[b,M,K] @ B[b,K,N]。每个 batch 走一次已验收的 sgemm。"""
        if len(A.shape) != 3 or len(B.shape) != 3:
            raise CudaError("bmm 需要 3D，收到 %s @ %s" % (A.shape, B.shape))
        batch, M, K = (int(s) for s in A.shape)
        batch2, K2, N = (int(s) for s in B.shape)
        if batch != batch2:
            raise CudaError("bmm batch 不同: %s @ %s" % (A.shape, B.shape))
        if K != K2:
            raise CudaError("bmm K 不匹配: %s @ %s" % (A.shape, B.shape))
        if batch == 0 or M == 0 or N == 0:
            return self.empty((batch, M, N), dtype=np.float32)
        if K == 0:
            return self.zeros((batch, M, N), dtype=np.float32)
        C = self.empty((batch, M, N), dtype=np.float32)
        a_bytes = M * K * 4
        b_bytes = K * N * 4
        c_bytes = M * N * 4
        for i in range(batch):
            self._sgemm(
                CUDA_OP_N, CUDA_OP_N, N, M, K,
                B.ptr + i * b_bytes, N,
                A.ptr + i * a_bytes, K,
                C.ptr + i * c_bytes, N,
            )
        return C

    def softmax_lastdim(self, x):
        if len(x.shape) < 1:
            raise CudaError("softmax 需要至少 1 维，收到 %s" % (x.shape,))
        k = int(x.shape[-1])
        if k <= 0:
            raise CudaError("softmax 最后一维是 %d" % k)
        rows = int(x.n) // k
        out = self.empty(x.shape, dtype=np.float32)
        if rows:
            grid, block = self._ewise_grid(rows)
            self.launch("softmax_lastdim", grid, block, [
                self._ptr(x), self._ptr(out),
                (rows, ctypes.c_int), (k, ctypes.c_int),
            ])
        return out

    def log_softmax_lastdim(self, x):
        if len(x.shape) < 1:
            raise CudaError("log_softmax 需要至少 1 维，收到 %s" % (x.shape,))
        k = int(x.shape[-1])
        if k <= 0:
            raise CudaError("log_softmax 最后一维是 %d" % k)
        rows = int(x.n) // k
        out = self.empty(x.shape, dtype=np.float32)
        if rows:
            grid, block = self._ewise_grid(rows)
            self.launch("log_softmax_lastdim", grid, block, [
                self._ptr(x), self._ptr(out),
                (rows, ctypes.c_int), (k, ctypes.c_int),
            ])
        return out

    def log_softmax_lastdim_bwd(self, g, y):
        k = int(y.shape[-1])
        rows = int(y.n) // k if k else 0
        out = self.empty(y.shape, dtype=np.float32)
        if rows and k:
            grid, block = self._ewise_grid(rows)
            self.launch("log_softmax_lastdim_bwd", grid, block, [
                self._ptr(g), self._ptr(y), self._ptr(out),
                (rows, ctypes.c_int), (k, ctypes.c_int),
            ])
        return out

    def softmax_lastdim_bwd(self, g, y):
        k = int(y.shape[-1])
        rows = int(y.n) // k if k else 0
        out = self.empty(y.shape, dtype=np.float32)
        if rows and k:
            grid, block = self._ewise_grid(rows)
            self.launch("softmax_lastdim_bwd", grid, block, [
                self._ptr(g), self._ptr(y), self._ptr(out),
                (rows, ctypes.c_int), (k, ctypes.c_int),
            ])
        return out

    def log32(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_log32", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def exp32(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_exp32", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def exp(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_exp", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def log(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_log", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def recip(self, a):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_recip", grid, block, [
                self._ptr(a), self._ptr(out), (int(a.n), ctypes.c_int),
            ])
        return out

    def scale(self, a, scalar):
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_scale", grid, block, [
                self._ptr(a), (float(scalar), ctypes.c_float), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def _sgemm(self, transa, transb, m, n, k, A, lda, B, ldb, C, ldc):
        if m == 0 or n == 0 or k == 0:
            return
        alpha = ctypes.c_float(1.0)
        beta = ctypes.c_float(0.0)
        self._activate()
        self._chk_blas(self.cublas.cublasSgemm_v2(
            self.handle,
            int(transa), int(transb),
            int(m), int(n), int(k),
            ctypes.byref(alpha),
            int(A), int(lda),
            int(B), int(ldb),
            ctypes.byref(beta),
            int(C), int(ldc),
        ), "cublasSgemm_v2")

    def sgemm_compose(self, A, B, chunks, kind):
        M, K = int(A.shape[0]), int(A.shape[1])
        N = int(B.shape[1])
        ch = [int(x) for x in chunks][:8]
        while len(ch) < 8:
            ch.append(0)
        C = self.empty((M, N), dtype=np.float32)
        total = M * N
        if total and K:
            grid, block = self._ewise_grid(total)
            self.launch("sgemm_compose", grid, block, [
                self._ptr(A), self._ptr(B), self._ptr(C),
                (M, ctypes.c_int), (N, ctypes.c_int), (K, ctypes.c_int),
                (len([x for x in chunks if int(x) > 0]), ctypes.c_int),
                (int(kind), ctypes.c_int),
            ] + [(int(x), ctypes.c_int) for x in ch])
        return C

    def sgemm_mode(self, A, B, mode):
        """对照用：同一种形状的多种累加。mode 见 sgemm_mode kernel。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("sgemm_mode 需要 2D")
        M, K = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("sgemm_mode K 不匹配")
        if K > 32 and int(mode) == 3:
            raise CudaError("pairwise mode 只支持 K<=32")
        C = self.empty((M, N), dtype=np.float32)
        total = M * N
        if total and K:
            grid, block = self._ewise_grid(total)
            self.launch("sgemm_mode", grid, block, [
                self._ptr(A), self._ptr(B), self._ptr(C),
                (M, ctypes.c_int), (N, ctypes.c_int), (K, ctypes.c_int),
                (int(mode), ctypes.c_int),
            ])
        elif total:
            return self.full((M, N), 0.0)
        return C

    def _numpy_exact_gemm(self, M, N, K):
        """这些尺寸上 NumPy 1.26 / OpenBLAS 的 float32 matmul 就是顺序 FMA。

        实测 K<=16、M,N<=64 时 sgemm_mode 1 与 `@` 逐位相同。
        cublasSgemm_v2 即使 PEDANTIC 也有约 3e-8，Muon 五轮迭代后超过 1e-7。
        更大的矩阵仍走 cuBLAS，避免朴素 kernel 拖垮以后的训练。
        """
        return 0 <= M <= 64 and 0 <= N <= 64 and 0 <= K <= 16

    def sgemm(self, A, B):
        """行优先 C[M,N] = A[M,K] @ B[K,N]。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("sgemm 需要 2D，收到 %s @ %s" % (A.shape, B.shape))
        M, K = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("sgemm K 不匹配: %s @ %s" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        if self._numpy_exact_gemm(M, N, K):
            return self.sgemm_mode(A, B, 1)
        C = self.empty((M, N), dtype=np.float32)
        # 列主序视角: C^T(N,M) = B^T(op=N, ld=N) @ A^T(op=N, ld=K)
        self._sgemm(CUDA_OP_N, CUDA_OP_N, N, M, K, B.ptr, N, A.ptr, K, C.ptr, N)
        return C

    def gemm_nt(self, A, B):
        """行优先 C[M,N] = A[M,K] @ B[N,K]^T。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("gemm_nt 需要 2D，收到 %s @ %s^T" % (A.shape, B.shape))
        M, K = int(A.shape[0]), int(A.shape[1])
        N, K2 = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("gemm_nt K 不匹配: %s @ %s^T" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        if self._numpy_exact_gemm(M, N, K):
            return self.sgemm_mode(A, self.transpose2d(B), 1)
        C = self.empty((M, N), dtype=np.float32)
        self._sgemm(CUDA_OP_T, CUDA_OP_N, N, M, K, B.ptr, K, A.ptr, K, C.ptr, N)
        return C

    def gemm_tn(self, A, B):
        """行优先 C[M,N] = A[K,M]^T @ B[K,N]。"""
        if len(A.shape) != 2 or len(B.shape) != 2:
            raise CudaError("gemm_tn 需要 2D，收到 %s^T @ %s" % (A.shape, B.shape))
        K, M = int(A.shape[0]), int(A.shape[1])
        K2, N = int(B.shape[0]), int(B.shape[1])
        if K != K2:
            raise CudaError("gemm_tn K 不匹配: %s^T @ %s" % (A.shape, B.shape))
        if M == 0 or N == 0 or K == 0:
            return self.full((M, N), 0.0)
        if self._numpy_exact_gemm(M, N, K):
            return self.sgemm_mode(self.transpose2d(A), B, 1)
        C = self.empty((M, N), dtype=np.float32)
        self._sgemm(CUDA_OP_N, CUDA_OP_T, N, M, K, B.ptr, N, A.ptr, M, C.ptr, N)
        return C

    def ce_fwd(self, logits, targets, n, v):
        n = int(n)
        v = int(v)
        loss = self.empty((), dtype=np.float32)
        if n <= 0 or v <= 0:
            self._memset32(loss, 0)
            loss.known_scalar = 0.0
            return loss
        terms = self.empty((n,), dtype=np.float32)
        self.launch("ce_fwd_rows", (n, 1, 1), (256, 1, 1), [
            self._ptr(logits), self._ptr(targets), self._ptr(terms),
            (n, ctypes.c_int), (v, ctypes.c_int),
        ])
        self.launch("ce_fwd_mean", (1, 1, 1), (1, 1, 1), [
            self._ptr(terms), self._ptr(loss), (n, ctypes.c_int),
        ])
        # terms 还被 ce_fwd_mean 读着。挂在 loss 上，避免函数返回时 free() 把整条流同步掉。
        loss.keepalive = terms
        return loss

    def ce_bwd(self, logits, targets, n, v):
        n = int(n)
        v = int(v)
        dl = self.empty((n, v), dtype=np.float32)
        if n > 0 and v > 0:
            self.launch("ce_bwd", (n, 1, 1), (256, 1, 1), [
                self._ptr(logits), self._ptr(targets), self._ptr(dl),
                (n, ctypes.c_int), (v, ctypes.c_int),
            ])
        return dl

    def zeros(self, shape, dtype=np.float32):
        return self.full(shape, 0, dtype=dtype)

    def mul(self, a, b):
        if a.shape != b.shape or a.n != b.n:
            raise CudaError("ewise_mul 形状不同: %s vs %s" % (a.shape, b.shape))
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_mul", grid, block, [
                self._ptr(a), self._ptr(b), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def sum_all(self, buf):
        """全归约。一个 256 线程的 block，M2 的张量都很小。"""
        out = self.empty((), dtype=np.float32)
        n = int(buf.n)
        if n <= 0:
            zero = np.zeros((), dtype=np.float32)
            self._chk(self.cu.cuMemcpyHtoD(
                ctypes.c_uint64(out.ptr),
                zero.ctypes.data_as(ctypes.c_void_p),
                ctypes.c_size_t(4)), "cuMemcpyHtoD(sum0)")
            return out
        self.launch("reduce_sum", (1, 1, 1), (256, 1, 1), [
            self._ptr(buf), self._ptr(out), (n, ctypes.c_int),
        ])
        return out

    def rmsnorm_fwd(self, x, w, seq, d, eps):
        out = self.empty(x.shape, dtype=np.float32)
        if seq and d:
            # 动态共享内存按 256 个 float。block 必须是 256，和 kernel 里的树归约一致。
            self.launch("rmsnorm_fwd", (int(seq), 1, 1), (256, 1, 1), [
                self._ptr(out), self._ptr(x), self._ptr(w),
                (int(seq), ctypes.c_int), (int(d), ctypes.c_int),
                (float(eps), ctypes.c_float),
            ], shared=256 * 4)
        return out

    def rmsnorm_bwd(self, dy, x, w, seq, d, eps):
        dx = self.empty(x.shape, dtype=np.float32)
        dw = self.zeros(w.shape, dtype=np.float32)
        if seq and d:
            # rmsnorm_bwd 的静态 smem 是 256，block 不能改。
            self.launch("rmsnorm_bwd", (int(seq), 1, 1), (256, 1, 1), [
                self._ptr(dy), self._ptr(x), self._ptr(w),
                self._ptr(dx), self._ptr(dw),
                (int(seq), ctypes.c_int), (int(d), ctypes.c_int),
                (float(eps), ctypes.c_float),
            ])
        return dx, dw

    def silu_mul_fwd(self, a, b):
        """原地：a[i] = silu(a[i]) * b[i]。调用方要先 clone。"""
        if a.n != b.n:
            raise CudaError("silu_mul 长度不同: %s vs %s" % (a.n, b.n))
        if a.n:
            grid, block = self._ewise_grid(a.n)
            self.launch("silu_mul_fwd", grid, block, [
                self._ptr(a), self._ptr(b), (int(a.n), ctypes.c_int),
            ])
        return a

    def silu_mul_bwd(self, g, a, b):
        da = self.empty(a.shape, dtype=np.float32)
        db = self.empty(b.shape, dtype=np.float32)
        n = int(a.n)
        if n:
            grid, block = self._ewise_grid(n)
            self.launch("silu_mul_bwd", grid, block, [
                self._ptr(g), self._ptr(a), self._ptr(b),
                self._ptr(da), self._ptr(db), (n, ctypes.c_int),
            ])
        return da, db

    def emb_fwd(self, table, ids, seq, d, vocab, out_shape):
        out = self.empty(out_shape, dtype=np.float32)
        if seq and d:
            self.launch("emb_gather_fwd", (int(seq), 1, 1), (256, 1, 1), [
                self._ptr(out), self._ptr(table), self._ptr(ids),
                (int(seq), ctypes.c_int), (int(d), ctypes.c_int),
                (int(vocab), ctypes.c_int), (0, ctypes.c_int),
            ])
        return out

    def emb_bwd(self, grad, ids, seq, d, table_shape):
        dw = self.zeros(table_shape, dtype=np.float32)
        if seq and d:
            self.launch("emb_gather_bwd", (int(seq), 1, 1), (256, 1, 1), [
                self._ptr(grad), self._ptr(ids), self._ptr(dw),
                (int(seq), ctypes.c_int), (int(d), ctypes.c_int),
            ])
        return dw

    def adamw(self, p, g, m, v, b1, omb1, b2, omb2, lr, wd_lr, eps, bc1, bc2, do_wd):
        """原地更新参数和两份动量。标量必须已经按 numpy 的 float32 规则收好。"""
        n = int(p.n)
        if int(g.n) != n or int(m.n) != n or int(v.n) != n:
            raise CudaError(
                "adamw 长度不一致: p=%d g=%d m=%d v=%d" % (n, g.n, m.n, v.n))
        if n <= 0:
            return
        if n > 2147483647:
            raise CudaError("adamw 元素个数超过 int32")
        grid, block = self._ewise_grid(n)
        self.launch("adamw_update", grid, block, [
            self._ptr(p), self._ptr(g), self._ptr(m), self._ptr(v),
            (n, ctypes.c_int),
            (float(b1), ctypes.c_float), (float(omb1), ctypes.c_float),
            (float(b2), ctypes.c_float), (float(omb2), ctypes.c_float),
            (float(lr), ctypes.c_float), (float(wd_lr), ctypes.c_float),
            (float(eps), ctypes.c_float),
            (float(bc1), ctypes.c_float), (float(bc2), ctypes.c_float),
            (1 if do_wd else 0, ctypes.c_int),
        ])

    def div_scalar(self, a, scalar):
        """c = a / scalar。真除法，不是乘倒数。"""
        out = self.empty(a.shape, dtype=np.float32)
        if a.n:
            if int(a.n) > 2147483647:
                raise CudaError("div_scalar 元素个数超过 int32")
            grid, block = self._ewise_grid(a.n)
            self.launch("ewise_div_scalar", grid, block, [
                self._ptr(a), (float(scalar), ctypes.c_float), self._ptr(out),
                (int(a.n), ctypes.c_int),
            ])
        return out

    def sub_scale_(self, p, o, scale):
        """原地 p = p - scale * o。"""
        n = int(p.n)
        if int(o.n) != n:
            raise CudaError("sub_scale 长度不一致: p=%d o=%d" % (n, o.n))
        if n <= 0:
            return p
        if n > 2147483647:
            raise CudaError("sub_scale 元素个数超过 int32")
        grid, block = self._ewise_grid(n)
        self.launch("ewise_sub_scale", grid, block, [
            self._ptr(p), self._ptr(o), (float(scale), ctypes.c_float),
            (n, ctypes.c_int),
        ])
        return p

    def muon_momentum(self, p, g, buf, u, mom, wd_lr, do_wd, nesterov):
        """原地写 p（weight decay）和 buf，u 是正交化的输入。"""
        n = int(p.n)
        if int(g.n) != n or int(buf.n) != n or int(u.n) != n:
            raise CudaError(
                "muon_momentum 长度不一致: p=%d g=%d buf=%d u=%d" % (n, g.n, buf.n, u.n))
        if n <= 0:
            return
        if n > 2147483647:
            raise CudaError("muon_momentum 元素个数超过 int32")
        grid, block = self._ewise_grid(n)
        self.launch("muon_momentum", grid, block, [
            self._ptr(p), self._ptr(g), self._ptr(buf), self._ptr(u),
            (n, ctypes.c_int),
            (float(mom), ctypes.c_float), (float(wd_lr), ctypes.c_float),
            (1 if do_wd else 0, ctypes.c_int),
            (1 if nesterov else 0, ctypes.c_int),
        ])

    def sumsq_sequential(self, buf):
        """一个标量：按地址顺序的 sum(x*x)。调用方只把这个标量降回主机。"""
        out = self.empty((), dtype=np.float32)
        n = int(buf.n)
        if n < 0 or n > 2147483647:
            raise CudaError("sumsq 元素个数非法: %d" % n)
        self.launch("sumsq_sequential", (1, 1, 1), (1, 1, 1), [
            self._ptr(buf), self._ptr(out), (n, ctypes.c_int),
        ])
        return out

    def _moe_row_block(self, d):
        d = int(d)
        if d <= 32:
            return 32
        if d <= 128:
            return 128
        return 256

    def moe_topk(self, logits, k):
        """每个 token 的 top-k。返回 (idx int32 [N,K], 选中 logit [N,K])。"""
        if len(logits.shape) != 2:
            raise CudaError("moe_topk 需要 [N,E]，收到 %s" % (logits.shape,))
        n, e = int(logits.shape[0]), int(logits.shape[1])
        k = int(k)
        if k < 1 or k > e:
            raise CudaError("moe_topk k=%d 超出专家数 %d" % (k, e))
        idx = self.empty((n, k), dtype=np.int32)
        val = self.empty((n, k), dtype=np.float32)
        if n and e:
            self.launch("moe_topk", (n, 1, 1), (1, 1, 1), [
                self._ptr(logits), self._ptr(idx), self._ptr(val),
                (n, ctypes.c_int), (e, ctypes.c_int), (k, ctypes.c_int),
            ])
        idx.keepalive = val
        return idx, val

    def moe_sigmoid(self, x):
        out = self.empty(x.shape, dtype=np.float32)
        if x.n:
            grid, block = self._ewise_grid(x.n)
            self.launch("moe_sigmoid", grid, block, [
                self._ptr(x), self._ptr(out), (int(x.n), ctypes.c_int),
            ])
        return out

    def moe_gather_pairs(self, src, idx, n, e, k):
        out = self.empty((int(n), int(k)), dtype=np.float32)
        if n and k:
            self.launch("moe_gather_pairs", (int(n), 1, 1), (32, 1, 1), [
                self._ptr(src), self._ptr(idx), self._ptr(out),
                (int(n), ctypes.c_int), (int(e), ctypes.c_int), (int(k), ctypes.c_int),
            ])
        return out

    def moe_scatter_pairs(self, g, idx, n, e, k):
        out = self.zeros((int(n), int(e)), dtype=np.float32)
        if n and e and k:
            self.launch("moe_scatter_pairs", (1, 1, 1), (1, 1, 1), [
                self._ptr(g), self._ptr(idx), self._ptr(out),
                (int(n), ctypes.c_int), (int(e), ctypes.c_int), (int(k), ctypes.c_int),
            ])
        return out

    def moe_compact(self, idx, n, e, k):
        """按专家把 (token, slot) 收成三段连续区间。offsets 长度 E+1。"""
        n, e, k = int(n), int(e), int(k)
        offsets = self.empty((e + 1,), dtype=np.int32)
        cursor = self.empty((max(e, 1),), dtype=np.int32)
        slots = n * k
        tok = self.empty((max(slots, 1),), dtype=np.int32)
        slot = self.empty((max(slots, 1),), dtype=np.int32)
        tok.n = slots
        slot.n = slots
        tok.shape = (slots,)
        slot.shape = (slots,)
        if e:
            self.launch("moe_compact", (1, 1, 1), (1, 1, 1), [
                self._ptr(idx), self._ptr(offsets), self._ptr(cursor),
                self._ptr(tok), self._ptr(slot),
                (n, ctypes.c_int), (e, ctypes.c_int), (k, ctypes.c_int),
            ])
        offsets.keepalive = (cursor, tok, slot, idx)
        return offsets, tok, slot

    def moe_gather_rows(self, x, tok, begin, m, d):
        out = self.empty((int(m), int(d)), dtype=np.float32)
        if m and d:
            self.launch("moe_gather_rows", (int(m), 1, 1), (self._moe_row_block(d), 1, 1), [
                self._ptr(x), self._ptr(tok), self._ptr(out),
                (int(begin), ctypes.c_int), (int(m), ctypes.c_int), (int(d), ctypes.c_int),
            ])
        return out

    def moe_scatter_rows(self, y, tok, begin, m, n, d):
        out = self.zeros((int(n), int(d)), dtype=np.float32)
        if m and d:
            self.launch("moe_scatter_rows", (int(m), 1, 1), (self._moe_row_block(d), 1, 1), [
                self._ptr(y), self._ptr(tok), self._ptr(out),
                (int(begin), ctypes.c_int), (int(m), ctypes.c_int), (int(d), ctypes.c_int),
            ])
        return out

    def moe_gather_slots(self, w, tok, slot, begin, m, k):
        out = self.empty((int(m),), dtype=np.float32)
        if m:
            grid, block = self._ewise_grid(m)
            self.launch("moe_gather_slots", grid, block, [
                self._ptr(w), self._ptr(tok), self._ptr(slot), self._ptr(out),
                (int(begin), ctypes.c_int), (int(m), ctypes.c_int), (int(k), ctypes.c_int),
            ])
        return out

    def moe_scatter_slots(self, g, tok, slot, begin, m, n, k):
        out = self.zeros((int(n), int(k)), dtype=np.float32)
        if m and n and k:
            self.launch("moe_scatter_slots", (1, 1, 1), (1, 1, 1), [
                self._ptr(g), self._ptr(tok), self._ptr(slot), self._ptr(out),
                (int(begin), ctypes.c_int), (int(m), ctypes.c_int),
                (int(n), ctypes.c_int), (int(k), ctypes.c_int),
            ])
        return out

    def moe_row_scale(self, y, w):
        if len(y.shape) != 2:
            raise CudaError("moe_row_scale 需要 2D，收到 %s" % (y.shape,))
        m, d = int(y.shape[0]), int(y.shape[1])
        if int(w.n) != m:
            raise CudaError("moe_row_scale 权重长度 %d != 行数 %d" % (w.n, m))
        out = self.empty((m, d), dtype=np.float32)
        if m and d:
            self.launch("moe_row_scale", (m, 1, 1), (self._moe_row_block(d), 1, 1), [
                self._ptr(y), self._ptr(w), self._ptr(out),
                (m, ctypes.c_int), (d, ctypes.c_int),
            ])
        return out

    def moe_row_dot(self, g, y):
        m, d = int(g.shape[0]), int(g.shape[1])
        out = self.empty((m,), dtype=np.float32)
        if m and d:
            self.launch("moe_row_dot", (m, 1, 1), (1, 1, 1), [
                self._ptr(g), self._ptr(y), self._ptr(out),
                (m, ctypes.c_int), (d, ctypes.c_int),
            ])
        elif m:
            self._memset32(out, 0)
        return out


_RT = None


def runtime():
    """进程内唯一运行时。第一次调用才加载 DLL。"""
    global _RT
    if _RT is None:
        _RT = CudaRuntime()
    return _RT.ensure()
