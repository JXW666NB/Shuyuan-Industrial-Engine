/* llama / qwen2 前向的 C 编排契约。同一份算子，差异只在 TgArch。
 *
 * Python 一次 forward 只进这个入口一次：句柄、权重设备指针、host ids 进来，
 * host logits 出去。层循环、发射、scratch 都留在 .so 里，不再逐算子回 Python。
 *
 * 本轮布局（行主序、float32），由描述符选中：
 *   RMSNorm(eps=1e-6) → [可选 Q/K/V bias] → RoPE(rotate-half, x@R) →
 *   GQA repeat → causal SDPA(scale=1/sqrt(dh)) → 残差 → RMSNorm →
 *   SwiGLU(silu(gate)*up) → 残差 → final RMSNorm → lm_head
 *
 * llama：qkv_bias=0。qwen2：qkv_bias=1，bias 加在投影之后、RoPE 之前。
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
    TG_POS_ROPE = 1,
    TG_FFN_SWIGLU = 1
};

typedef struct TgArch {
    int32_t qkv_bias; /* 0 llama：投影后不加 bias；1 qwen2：加 Q/K/V bias */
    int32_t norm;     /* TG_NORM_RMS */
    int32_t pos;      /* TG_POS_ROPE */
    int32_t ffn;      /* TG_FFN_SWIGLU */
    int32_t n_kv;     /* GQA 的 kv 头数；repeat = nh / n_kv */
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
    int32_t d;
    int32_t L;
    int32_t nh;
    int32_t ff;
    int32_t V;
    int32_t rope_seq;
    int32_t n_w;
    TgArch arch;
    /* host 指针，指向 n_w 个设备地址（uint64）。
     * 0 tok_emb [V,D]
     * 1 final_norm [D]
     * 2 lm_head [V,D]
     * 3 cos [rope_seq, dh]
     * 4 sin [rope_seq, dh]
     * 5 R [dh, dh]  rotate-half
     * 之后每层：attn_norm, q, k, v, o, ffn_norm, gate, up, down
     * qkv_bias=1 时这一层再接 q_bias[D], k_bias[n_kv*dh], v_bias[n_kv*dh]
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
