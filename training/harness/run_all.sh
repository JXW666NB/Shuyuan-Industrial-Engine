#!/bin/bash
# 全量回归（收尾用）—— 只打印每条的关键行
cd /home/featurize/tgai || exit 1
export PYTHONPATH=/home/featurize/tgai/python
H=training/harness
P=python3
line() { printf '%-34s %s\n' "$1" "$2"; }
grab() { grep -E "$2" | tail -1; }

echo "==== 判分台自证 ===="
line "conformance CPU"  "$($P -u $H/pytorch_conformance.py 2>&1 | grep '迁移一致性' | tail -1)"
line "conformance CUDA" "$($P -u $H/pytorch_conformance.py --cuda 2>&1 | grep '迁移一致性' | tail -1)"
for g in train_backend_parity train_op_parity train_layer_parity train_opt_parity; do
  line "$g selftest" "$($P -u $H/$g.py --selftest 2>&1 | grep VERDICT | tail -1)"
done
line "moe_selfcheck" "$($P -u $H/moe_selfcheck.py 2>&1 | grep VERDICT | tail -1)"
line "opt_step_parity cuda" "$($P -u $H/opt_step_parity.py --cuda 2>&1 | grep VERDICT | tail -1)"
line "attn_lowmem" "$($P -u $H/attn_lowmem_check.py 2>&1 | grep -c PASS) PASS / $($P -u $H/attn_lowmem_check.py 2>&1 | grep -c FAIL) FAIL"

echo "==== 预设架构：Python 与 C ===="
line "llama CPU"  "$($P -u $H/llama_parity.py --cpu 2>&1 | grep 'logits rel' | tail -1)"
line "llama CUDA" "$($P -u $H/llama_parity.py --cuda 2>&1 | grep 'logits rel' | tail -1)"
line "qwen2 CUDA" "$($P -u $H/llama_parity.py --arch qwen2 --cuda 2>&1 | grep 'logits rel' | tail -1)"
line "llama C"    "$($P -u $H/llama_parity.py --c-forward 2>&1 | grep 'logits rel' | tail -1)"
line "qwen2 C"    "$($P -u $H/llama_parity.py --c-forward --arch qwen2 2>&1 | grep 'logits rel' | tail -1)"
line "gpt2 CUDA"  "$($P -u $H/gpt2_parity.py --cuda 2>&1 | grep 'logits rel' | tail -1)"
line "c_arch(3 架构)" "$($P -u $H/c_arch_parity.py --compare 2>&1 | grep VERDICT | tail -1)"

echo "==== V3 ===="
line "v3 前向" "$($P -u $H/v3_parity.py --compare 2>&1 | grep '相对误差' | tail -1)"
line "v3 反向" "$($P -u $H/v3_grad_parity.py --compare 2>&1 | grep '最差:' | tail -1)"
line "v3 1/8 规模" "见 docs（1.062e-05）"

echo "==== 流水线 ===="
line "pipeline check" "$($P -u $H/pipeline_smoke.py --check 2>&1 | grep 'loss' | tail -1)"

echo "==== 性能 / 显存 ===="
line "seq2048" "$($P -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 2048 2>&1 | grep '吞吐' | tail -1)"
line "seq8192" "$($P -u training/harness/v27_moe_bench.py --backend tgai --layers 1 --steps 3 --seq 8192 2>&1 | grep '吞吐' | tail -1)"
line "mem2048" "$($P -u $H/a3_mem_parity.py --backend tgai --seq 2048 2>&1 | grep '峰值' | tail -1)"
line "mem8192" "$($P -u $H/a3_mem_parity.py --backend tgai --seq 8192 2>&1 | grep '峰值' | tail -1)"