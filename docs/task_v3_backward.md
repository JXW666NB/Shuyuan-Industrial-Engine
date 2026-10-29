# 任务：B4 反向 —— V3（非 CSA）的梯度与官方实现对齐
## 0. 纪律
- 本地 `D:\DSH-Workspace\TGAI-对话档案`，远端用 `feat.py sh/put`；
  包 `/home/featurize/tgai/python/tgai_go/`，判分台 `/home/featurize/tgai/training/harness/`。
- **不改 SSH key / 密码 / sudo**；本轮 CPU 就够，不必占 4090。
- 改文件前备份；结论必须有实测数字。

## 1. 现状（我已做完的部分，别重做）
- `training/harness/v3_ref.py`：**前向**已与官方对齐（相对误差 **5.021e-05**），
  但它用 numpy 做组装（MoE 的 gather/scatter、AttnRes 的 softmax 都是 numpy），**没有 autograd**。
- `training/oracle/v3_tiny_*`：官方实现导出的真值
  - `weights.npz`（131 个张量）、`ids.npy`
  - `grads.npz`（**112** 个参数梯度）、`upstream.npy`（上游权重 `(1,16,64)`）、loss=**4.653289**
- 判分台 `training/harness/v3_grad_parity.py` 已写好：`--selftest`（同侧两次必须 0）、
  `--compare`（逐参数比 <1e-3）。它调用 **`v3_ref.V3Auto`** —— 这个类还不存在，是你本轮要实现的。

## 2. 要做的事
在 `training/harness/v3_ref.py` 里新增 **`class V3Auto`**，与现有 `V3Forward` **前向语义完全一致**，
但**全程可求导**（用我们自己的 `tgai_go.train` 张量算子），暴露：

```python
class V3Auto:
    def __init__(self, cfg, weights): ...      # weights: {名字: ndarray}，转成 T.Parameter
    def parameters(self): ...                  # 返回 Parameter 列表
    def forward(self, ids):                   # -> logits，形状 [B, S, V]
    # self.w 是 {名字: T.Parameter}，名字与 oracle 完全一致（blocks.0.attn.q_proj.weight 等）
```

### 关键：把 numpy 那段改成可求导写法
1. **MoE 的按专家 gather/scatter**：numpy 索引会断梯度。两种可行做法：
   - 用 `T.nn.Embedding`/`__getitem__` + 可求导的 scatter（若已有），或
   - **掩码法**：对每个专家算**全部 token**（`experts[e](x_all)`），再用 0/1 掩码按 gate 加权求和
     —— 小配置（N=16 tokens、E=4）下代价可接受，且天然可求导。
2. **AttnRes 的 softmax 加权**：`alpha` 用 `T.functional.softmax`，加权求和用张量乘加（可求导）。
3. **top-k 选择**：索引本身不需求导（阶梯函数），但**门控权重**必须可求导
   （`gate = s_sel / Σ s_sel`，`s = sigmoid(router(x))`）。
4. **SiTU-GLU**、**QK-Norm**、**RoPE**、**RMSNorm** 全部用张量算子（前向已在 `V3Forward` 里验证过）。

## 3. 判据
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_grad_parity.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/v3_grad_parity.py --compare"
```
- 自证：同侧两次 **每个梯度逐位一致（0）**
- 考核：112 个梯度**全部给出**且相对误差 **< 1e-3**；loss 应等于 **4.653289**

## 4. 不许回归
`v3_parity.py --compare` 仍应 5e-5 量级；`llama_parity --cuda/--c-forward/--arch qwen2`、
`gpt2_parity --cuda`、`c_arch_parity --compare`、`pipeline_smoke.py --selftest/--check` 全绿；
`pytorch_conformance` CPU 13/13。

## 5. 交付
报告 `docs/train_v3_backward.md`：哪些算子改了写法（尤其 MoE gather/scatter）、
自证值、loss、最差 5 个梯度的相对误差。回话最后 6-8 行给这些数字。
