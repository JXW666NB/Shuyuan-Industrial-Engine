# 任务：B6 一键流水线（语料目录 -> 自训 BPE -> 预设 -> 训练 -> 存盘 -> 重载）
## 0. 纪律
- 本地 `D:\DSH-Workspace\TGAI-对话档案`，远端用 `feat.py sh/put`。
- 包 `/home/featurize/tgai/python/tgai_go/`；训练侧工具 `/home/featurize/tgai/training/`；
  判分台 `/home/featurize/tgai/training/harness/`。
- **不改 SSH key / 密码 / sudo**；单次 GPU 跑 ≤10 分钟。
- 本轮 CPU 就能验（不需要 4090），不必占 GPU。

## 1. 现状（可复用，别重写）
- `training/bpe_train.py`：`train_bpe(text, target_vocab)` 与 `encode_ref(text, vocab, merges)`（纯 Python BPE）
- 预设：`training/harness/llama_ref.py`（`make_cfg` / `Llama`）、`gpt2_ref.py`、`qwen2_ref.py`
- 导出：`python/tgai_go/train/export.py`（写 `.TG`；**本任务暂不要求**，见 §3）

## 2. 要做的事：新增 `python/tgai_go/train/pipeline.py`
对外只暴露一个函数（签名必须一致，判分台按这个调）：

```python
def run(corpus_dir, out_dir, arch="llama", vocab_size=512, seq=128, steps=200,
        seed=0, d_model=64, n_layers=1, n_heads=4, n_kv=2, d_ff=128,
        lr=3e-3, device="cpu"):
    """返回 dict，至少包含：
       first_loss, final_loss, steps, vocab_size, n_merges, reload_bitwise(bool)
    """
```

流程：
1. 读 `corpus_dir` 下所有 `*.txt`（UTF-8），拼成一段文本；
2. **自训 BPE**（用 `bpe_train.train_bpe`），词表与 merges 由流水线产生，**用户不提供分词表**；
3. 把语料编码成 token id（用 `encode_ref`），切成 `[B, seq]` 的样本；
4. 按 `arch` 建模型（本轮只需支持 `llama`，其余架构抛 `NotImplementedError`）；
5. AdamW 训练 `steps` 步，记录首末 loss；
6. 存盘到 `out_dir`：`tokenizer.json`（词表+merges）、`ckpt.npz`（参数）、`meta.json`（配置+loss）；
7. **重载校验**：从 `ckpt.npz` 重新加载参数，对同一批输入前向，要求与原 logits
   **逐位一致**（`np.array_equal`），结果写进返回值的 `reload_bitwise`。

## 3. 本轮不做（下一步）
- 写 `.TG`：`export.py` 已存在，但引擎侧有内存安全问题待修，先不接进流水线；
- 真机大规模训练：本轮只要求"小配置能跑通、loss 下降、重载一致"。

## 4. 判据（判分台已写好）
```
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/pipeline_smoke.py --selftest"
python feat.py sh "cd /home/featurize/tgai && PYTHONPATH=/home/featurize/tgai/python python3 -u training/harness/pipeline_smoke.py --check"
```
- 自证：**同 seed 两次 final_loss 差必须恰好 0**
- 验收：logits 重载后**逐位一致**；loss 末值 < 首值×0.8；三个产出物齐全

## 5. 不许回归
`pytorch_conformance` CPU 13/13；四 `--selftest` 全 0；`llama_parity --cuda`/`--c-forward`、
`gpt2_parity --cuda`、`c_arch_parity --compare` 仍绿（这些都是 CPU/GPU 轻量项）。

## 6. 交付
报告 `docs/train_pipeline_b6.md`：函数签名、流程、BPE 自训用了多少词/多少 merges、loss 曲线首末、重载校验结果。
回话最后 6-8 行：自证是否 0、loss 首末、重载是否逐位一致、回归是否全过。
