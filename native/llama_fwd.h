/* llama / qwen2 / gpt2 前向的 C 编排契约。同一份算子，差异只在 TgArch。
 *
 * Python 一次 forward 只进这个入口一次：句柄、权重设备指针、host ids 进来，
 * host logits 出去。层循环、发射、scratch 都留在 .so 里，不再逐算子回 Python。
 *
 * 由描述符选中（行主序、float32）：
 *   norm：RMSNorm(eps=1e-6) 或 LayerNorm(gamma+beta, eps=1e-5)
 *   pos ：RoPE(rotate-half, x@R) 或学习式 wpe（加在 token embedding 上）
 *   ffn ：SwiGLU(silu(gate)*up) 或 GELU-tanh 的两层 MLP
 *   然后 GQA repeat → causal SDPA(scale=1/sqrt(dh)) → 残差 → final norm → lm_head
 *
 * llama：qkv_bias=0，RMS + RoPE + SwiGLU。
 * qwen2：qkv_bias=1，bias 加在 Q/K/V 投影之后、RoPE 之前。
 * gpt2：bias_all=1，LayerNorm + wpe + GELU，q/k/v/o/fc/proj 都带 bias。
 *
 * LayerNorm 和 GELU 是共享原语（sum/pow/exp/broadcast 拼出来），不是另一份算子树。
 */
#ifndef TGAI_LLAMA_FWD_H
#define TGAI_LLAMA_FWD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 与 Python presets._Arch / _Plan 逐字段对齐。改这里必须同步 presets.py。
 * 枚举从 1 起：全 0 的描述符直接拒绝，不会被当成 llama。
 */
enum {
    TG_NORM_RMS = 1,
    TG_NORM_LN = 2,       /* LayerNorm，gamma + beta，eps=1e-5 */
    TG_POS_ROPE = 1,
    TG_POS_WPE = 2,       /* 学习式位置，按位置加 wpe */
    TG_FFN_SWIGLU = 1,
    TG_FFN_GELU = 2       /* GELU(tanh 近似) 的 fc → proj */
};

typedef struct TgArch {
    int32_t qkv_bias; /* 0：Q/K/V 不加 bias；1：只加 Q/K/V bias（qwen2） */
    int32_t norm;     /* TG_NORM_RMS 或 TG_NORM_LN */
    int32_t pos;      /* TG_POS_ROPE 或 TG_POS_WPE */
    int32_t ffn;      /* TG_FFN_SWIGLU 或 TG_FFN_GELU */
    int32_t n_kv;     /* GQA 的 kv 头数；repeat = nh / n_kv。gpt2 等于 nh */
    int32_t bias_all; /* 0 或 1。1：q/k/v/o 以及 FFN 的每个线性层都加 bias */
} TgArch;

typedef struct TgaiLlamaPlan {
    uint64_t cu_launch;
    uint64_t cu_h2d;
    uint64_t cu_d2h;
    uint64_t cu_d2d;
    uint64_t cu_alloc;
    uint64_t cu_free;
    uint64_t cu_memset32;
    uint64_t cu_stream_sync;
    uint64_t sgemm;
    uint64_t cublas;
    uint64_t k_emb;
    uint64_t k_rms;
    uint64_t k_perm;
    uint64_t k_add;
    uint64_t k_mul;
    uint64_t k_scale;
    uint64_t k_bcast;
    uint64_t k_softmax;
    uint64_t k_swap;
    uint64_t k_silu;
    uint64_t k_gather;
    uint64_t k_bcast_to;
    uint64_t k_exp;    /* ewise_exp，GELU 的 exp */
    uint64_t k_recip;  /* ewise_recip，留给共享层；本轮 GELU 用 pow(-1) */
    uint64_t k_pow;    /* ewise_pow，LayerNorm 的 rsqrt 与 GELU 的倒数 */
    uint64_t k_sum;    /* sum_axis8，LayerNorm 的 mean/var */
    int32_t d;
    int32_t L;
    int32_t nh;
    int32_t ff;
    int32_t V;
    int32_t rope_seq;
    int32_t n_w;
    TgArch arch;
    /* host 指针，指向 n_w 个设备地址（uint64）。条数必须和描述符算出来的一样。
     *
     * 头：
     *   0 tok / wte [V,D]
     *   1 final norm gamma [D]
     *   2 lm_head [V,D]
     *   pos=RoPE：cos [rope_seq,dh], sin [rope_seq,dh], R [dh,dh]
     *   pos=wpe ：wpe [rope_seq,D]
     *   norm=LayerNorm：再接 final beta [D]
     *
     * 每层，按这个顺序，缺的字段整段跳过：
     *   attn gamma [, attn beta], q, k, v, o, ffn gamma [, ffn beta]
     *   SwiGLU：gate, up, down
     *   GELU ：fc, proj
     *   qkv_bias 或 bias_all：q_bias[D], k_bias[n_kv*dh], v_bias[n_kv*dh]
     *   bias_all 再接 o_bias[D]，以及 FFN 每个线性层的 bias
     *     SwiGLU：gate_bias, up_bias, down_bias
     *     GELU ：fc_bias[ff], proj_bias[D]
     *
     * llama 因此仍是 6 + L*9。qwen2 仍是 6 + L*12。gpt2 是 5 + L*16。
     */
    uint64_t w;
} TgaiLlamaPlan;

/* 返回计划结构的字节数，Python 用来拒绝布局漂移。 */
int tgai_llama_plan_size(void);

/* 最近一次 tgai_llama_forward 里 Python→C 的进入次数（含本次）。失败也记。 */
int tgai_llama_last_ffi_calls(void);

/* 最近一次失败的短说明；成功时为空串。 */
const char* tgai_llama_last_error(void);

/* ids: host int64 [B,S] 行主序。logits: host float32 [B*S, V]。
 * 返回 0 成功，非 0 失败（看 last_error）。
 */
int tgai_llama_forward(
    const TgaiLlamaPlan* plan,
    const int64_t* ids,
    int32_t B,
    int32_t S,
    float* logits);

#ifdef __cplusplus
}
#endif

#endif
