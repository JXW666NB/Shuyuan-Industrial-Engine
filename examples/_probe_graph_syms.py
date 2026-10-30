# -*- coding: utf-8 -*-
import ctypes
cu = ctypes.CDLL("libcuda.so.1")
names = [
    "cuStreamCreate", "cuStreamBeginCapture", "cuStreamBeginCapture_v2",
    "cuStreamEndCapture", "cuGraphInstantiate", "cuGraphInstantiateWithFlags",
    "cuGraphLaunch", "cuGraphDestroy", "cuGraphExecDestroy", "cuGraphGetNodes",
    "cuGraphNodeGetType", "cuGraphKernelNodeGetParams",
    "cuGraphExecKernelNodeSetParams", "cuMemcpyDtoDAsync", "cuMemcpyDtoDAsync_v2",
]
for n in names:
    print("%s %s" % (n, hasattr(cu, n)))
