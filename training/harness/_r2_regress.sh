#!/bin/bash
set -u
cd /home/featurize/tgai
export PYTHONPATH=/home/featurize/tgai/python
run() {
  echo "===== $* ====="
  python3 -u "$@"
  echo "EXIT:$?"
}
run training/harness/train_backend_parity.py --selftest
run training/harness/train_op_parity.py --selftest
run training/harness/train_layer_parity.py --selftest
run training/harness/train_opt_parity.py --selftest
run training/harness/pytorch_conformance.py
run training/harness/train_backend_parity.py --cuda
run training/harness/train_op_parity.py --cuda
run training/harness/train_layer_parity.py --cuda
run training/harness/moe_selfcheck.py
run training/harness/opt_step_parity.py --cuda
run training/harness/attn_lowmem_check.py
run training/harness/pytorch_conformance.py --cuda
echo ALL_DONE
