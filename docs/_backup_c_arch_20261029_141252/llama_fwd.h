/* llama 前向的 C 编排契约。
 *
 * Python 一次 forward 只进这个入口一次：句柄、权重设备指针、host ids 进来，
 * host logits 出去。层循环、发射、scratch 都留在 .so 里，不再逐算子回 Python。
 *
 * 布局与 llama_ref.py 一致（行主序、float32）：
 *   RMSNorm(eps=1e-6) → RoPE(rotate-half, x@R) → GQA repeat →
 *   causal SDPA(scale=1/sqrt(dh)) → 残差 → RMSNorm →
 *   SwiGLU(silu(gate)*up) → 残差 → final RMSNorm → lm_head
 */
#ifndef TGAI_LLAMA_FWD_H
#define TGAI_LLAMA_FWD_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* 与 Python ctypes 结构逐字段对齐。改这里必须同步 presets.py。 */
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
    int32_t n_kv;
    int32_t ff;
    int32_t V;
    int32_t rope_seq;
    int32_t n_w;
    /* host 指针，指向 n_w 个设备地址（uint64）。
     * 0 tok_emb [V,D]
     * 1 final_norm [D]
     * 2 lm_head [V,D]
     * 3 cos [rope_seq, dh]
     * 4 sin [rope_seq, dh]
     * 5 R [dh, dh]  rotate-half
     * 之后每层 9 个：attn_norm, q, k, v, o, ffn_norm, gate, up, down
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
