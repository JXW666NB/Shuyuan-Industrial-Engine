# -*- coding: utf-8 -*-
"""用真引擎（libtgai_engine_gpu.dll）加载我们写出的 .TG，验证能前向。

判据:
  1. tg_load 成功
  2. header 里各维度与我们写入的一致
  3. tg_forward 返回 [seq, vocab] 的有限 logits

用法: python tg_load_test.py <file.tg> [dll路径]
"""
import ctypes
import os
import sys

import numpy as np

DEFAULT_DLL = r"D:\TGAI\libtgai_engine_gpu.dll"


class TGHeader(ctypes.Structure):
    _pack_ = 1
    _fields_ = [
        ("magic", ctypes.c_uint32), ("version", ctypes.c_uint32),
        ("weight_dtype", ctypes.c_uint32), ("n_layers", ctypes.c_uint32),
        ("d_model", ctypes.c_uint32), ("n_heads", ctypes.c_uint32),
        ("d_ff", ctypes.c_uint32), ("vocab_size", ctypes.c_uint32),
        ("max_seq_len", ctypes.c_uint32), ("n_experts", ctypes.c_uint32),
        ("n_activated", ctypes.c_uint32), ("rope_theta", ctypes.c_uint32),
        ("pad_token_id", ctypes.c_uint32), ("bos_token_id", ctypes.c_uint32),
        ("eos_token_id", ctypes.c_uint32), ("unk_token_id", ctypes.c_uint32),
        ("tokenizer_off", ctypes.c_uint64), ("tokenizer_len", ctypes.c_uint64),
        ("weight_table_off", ctypes.c_uint64), ("weight_table_count", ctypes.c_uint64),
        ("weight_data_off", ctypes.c_uint64), ("weight_data_len", ctypes.c_uint64),
        ("arch_type", ctypes.c_uint32), ("num_kv_heads", ctypes.c_uint32),
        ("modality_flags", ctypes.c_uint32), ("extension_off", ctypes.c_uint32),
    ]


def main(path, dll_path=DEFAULT_DLL):
    assert os.path.exists(path), path
    assert os.path.exists(dll_path), dll_path
    lib = ctypes.CDLL(dll_path)

    lib.tg_load.restype = ctypes.c_void_p
    lib.tg_load.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p]
    lib.tg_config.restype = ctypes.POINTER(TGHeader)
    lib.tg_config.argtypes = [ctypes.c_void_p]
    lib.tg_forward.restype = ctypes.POINTER(ctypes.c_float)
    lib.tg_forward.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int32),
                               ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
    lib.tg_last_error.restype = ctypes.c_char_p
    lib.tg_free.argtypes = [ctypes.c_void_p]

    err = ctypes.c_int(0)
    m = lib.tg_load(path.encode("utf-8"), 0, ctypes.byref(err))
    print("tg_load -> %s (err=%d)" % ("OK" if m else "NULL", err.value))
    if not m:
        print("  tg_last_error:", (lib.tg_last_error() or b"").decode("utf-8", "replace"))
        return 1

    h = lib.tg_config(m).contents
    print("=== 引擎读到的 header ===")
    print("  magic=0x%08X ver=%d dtype=%d L=%d d=%d nh=%d ff=%d V=%d seq=%d E=%d topk=%d"
          % (h.magic, h.version, h.weight_dtype, h.n_layers, h.d_model, h.n_heads,
             h.d_ff, h.vocab_size, h.max_seq_len, h.n_experts, h.n_activated))
    print("  pad/bos/eos/unk=%d/%d/%d/%d  arch=%d nkv=%d ext_off=%d"
          % (h.pad_token_id, h.bos_token_id, h.eos_token_id, h.unk_token_id,
             h.arch_type, h.num_kv_heads, h.extension_off))

    seq_len = min(4, h.max_seq_len)
    ids = (ctypes.c_int32 * seq_len)(*[1, 2, 3, 4][:seq_len])
    ptr = lib.tg_forward(m, ids, seq_len, None, 0)
    if not ptr:
        print("tg_forward -> NULL；tg_last_error:",
              (lib.tg_last_error() or b"").decode("utf-8", "replace"))
        lib.tg_free(m)
        return 1

    # ★ tg_forward 只返回最后一个位置的 logits（长度 V），不是 [seq, V]
    V = h.vocab_size
    arr = np.ctypeslib.as_array(ptr, shape=(V,)).reshape(1, V).copy()
    print("=== 前向 ===")
    print("  输出形状:", arr.shape)
    print("  finite=%s  min=%.4f max=%.4f mean=%.4f"
          % (bool(np.isfinite(arr).all()), arr.min(), arr.max(), arr.mean()))
    print("  末位 logits 前 8 个:", np.round(arr[-1, :8], 4).tolist())

    ok = (h.magic == 0x49414754 and arr.shape == (seq_len, V)
          and np.isfinite(arr).all())
    lib.tg_free(m)
    print("\nVERDICT ok=%s" % ok)
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else DEFAULT_DLL))
