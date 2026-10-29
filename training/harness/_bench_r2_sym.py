import ctypes
c = ctypes.CDLL("libcublas.so.12")
print("strided", hasattr(c, "cublasSgemmStridedBatched"))
