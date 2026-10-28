# -*- coding: utf-8 -*-
import inspect
import tgai_go.train as t
src = inspect.getsource(t._silu32_pair)
print("chain", "ds_dneg" in src)
print(src)
