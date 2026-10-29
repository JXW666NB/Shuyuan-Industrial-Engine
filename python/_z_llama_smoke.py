import sys
sys.path.insert(0, r"D:\DSH-Workspace\TGAI-对话档案\_repos\Shuyuan-Industrial-Engine\training\harness")
import numpy as np
import tgai_go.train as T
import llama_ref

cfg = llama_ref.make_cfg()
m = llama_ref.Llama(cfg, cfg["weights"])
g = np.random.RandomState(1)
ids = g.randint(0, cfg["V"], size=(2, cfg["seq"])).astype(np.int64)
tgt = g.randint(0, cfg["V"], size=2 * cfg["seq"]).astype(np.int64)
logits = m.forward(ids)
print("logits shape:", logits.shape)
loss = T.functional.cross_entropy(logits, tgt)
print("loss:", float(np.asarray(loss.to_numpy())))
loss.backward()
print("grad 样例:", np.asarray(m.w["blocks.0.attn.q.weight"].grad.to_numpy()).shape,
      float(np.abs(np.asarray(m.w["blocks.0.attn.q.weight"].grad.to_numpy())).max()))
