import ctypes
import numpy as np
lib = ctypes.CDLL(np.__config__.blas_opt_info["libraries"][0] if False else
                  "/environment/miniconda3/lib/python3.11/site-packages/numpy.libs/libopenblas64_p-r0-0cf96a72.3.23.dev.so",
                  mode=ctypes.RTLD_GLOBAL)
lib.openblas_get_corename64_.restype = ctypes.c_char_p
print(lib.openblas_get_corename64_())
lib.gotoblas_corename.restype = ctypes.c_char_p
print("goto", lib.gotoblas_corename())
