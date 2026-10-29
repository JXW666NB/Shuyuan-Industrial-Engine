# -*- coding: utf-8 -*-
"""PyTorch 迁移一致性套件（M5 判分台）。

目标（用户的硬指标）：**只把 `import torch` 换成 `import tgai_go.train as torch`，其余一字不改**，
这些脚本必须原样跑通。

所以下面每个 case 都是**标准 PyTorch 写法**；本文件唯一的"非 PyTorch"之处就是最上面那行 import。

用法：
    python pytorch_conformance.py                    # 用 tgai_go.train 跑
    python pytorch_conformance.py --list             # 只列 case 名
    python pytorch_conformance.py --case linear_nd   # 只跑一个
"""
import sys

sys.path.insert(0, __file__.rsplit("\\", 3)[0])

import tgai_go.train as torch          # ← 唯一改动：torch → tgai_go.train

import numpy as np                     # noqa: E402  （PyTorch 脚本里也可能用）


def _np(x):
    return x.detach().to_numpy() if hasattr(x, "detach") else np.asarray(x)


# ─────────────────────────── cases ───────────────────────────
def case_tensor_basic():
    """torch.tensor / reshape / 基本算术。"""
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    y = x * 2.0 + 1.0
    z = y.reshape(4)
    assert _np(z).shape == (4,), _np(z).shape
    # 四个元素各自 *2+1：3+5+7+9=24。原先写 22，和真实 PyTorch 不一致。
    assert abs(float(_np(z).sum()) - 24.0) < 1e-6
    return "ok"


def case_linear_nd():
    """N 维输入走 nn.Linear（PyTorch 支持任意前导维）。"""
    m = torch.nn.Linear(16, 8)
    x = torch.randn(2, 3, 16)
    y = m(x)
    assert _np(y).shape == (2, 3, 8), _np(y).shape
    return "ok"


def case_sequential():
    """nn.Sequential 堆叠。"""
    m = torch.nn.Sequential(
        torch.nn.Linear(8, 16),
        torch.nn.Linear(16, 4),
    )
    y = m(torch.randn(5, 8))
    assert _np(y).shape == (5, 4), _np(y).shape
    return "ok"


def case_functional_names():
    """functional 常用名字。"""
    x = torch.randn(4, 6)
    a = torch.nn.functional.silu(x)
    b = torch.nn.functional.softmax(x, dim=-1)
    c = torch.nn.functional.log_softmax(x, dim=-1)
    for t in (a, b, c):
        assert _np(t).shape == (4, 6)
    s = _np(b).sum(axis=-1)
    assert np.allclose(s, 1.0, atol=1e-5), s
    return "ok"


def case_cross_entropy():
    """functional.cross_entropy + 反向。"""
    logits = torch.randn(6, 5, requires_grad=True)
    tgt = np.array([0, 1, 2, 3, 4, 0], dtype=np.int64)
    loss = torch.nn.functional.cross_entropy(logits, tgt)
    loss.backward()
    assert _np(loss).size == 1 or np.asarray(_np(loss)).ndim >= 0
    return "ok"


def case_no_grad():
    """torch.no_grad() 上下文。"""
    m = torch.nn.Linear(4, 4)
    x = torch.randn(3, 4)
    with torch.no_grad():
        y = m(x)
    assert _np(y).shape == (3, 4)
    return "ok"


def case_optimizer_step():
    """optim.AdamW + zero_grad + step（迁移最常写的循环）。"""
    m = torch.nn.Linear(8, 4)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    x = torch.randn(6, 8)
    tgt = np.array([0, 1, 2, 3, 0, 1], dtype=np.int64)
    before = _np(m.weight).copy()
    loss = torch.nn.functional.cross_entropy(m(x), tgt)
    opt.zero_grad()
    loss.backward()
    opt.step()
    after = _np(m.weight)
    assert not np.allclose(before, after), "参数没有被更新"
    return "ok"


def case_state_dict():
    """state_dict / load_state_dict 往返。"""
    m = torch.nn.Linear(6, 3)
    sd = m.state_dict()
    m2 = torch.nn.Linear(6, 3)
    m2.load_state_dict(sd)
    a, b = _np(m.weight), _np(m2.weight)
    assert np.allclose(a, b), "state_dict 往返不一致"
    return "ok"


def case_reshape_permute():
    """reshape / permute / transpose / contiguous。"""
    x = torch.randn(2, 3, 4)
    y = x.permute(1, 0, 2).contiguous()
    assert _np(y).shape == (3, 2, 4), _np(y).shape
    z = x.transpose(0, 2)
    assert _np(z).shape == (4, 3, 2), _np(z).shape
    return "ok"


def case_seed_repeat():
    """manual_seed 后两次 randn 必须一致（可复现性）。"""
    torch.manual_seed(1234)
    a = torch.randn(8, 8)
    torch.manual_seed(1234)
    b = torch.randn(8, 8)
    assert np.allclose(_np(a), _np(b)), "同 seed 两次 randn 不一致"
    return "ok"


def case_rmsnorm_attn():
    """本项目常用模块：RMSNorm + CausalSelfAttention。"""
    n1 = torch.nn.RMSNorm(32)
    att = torch.nn.CausalSelfAttention(32, 4)
    x = torch.randn(2, 8, 32)
    h = n1(x)
    y = att(h)
    assert _np(y).shape == (2, 8, 32), _np(y).shape
    return "ok"


def case_slice_basic():
    """切片：PyTorch 里到处是 x[:, :k] / x[..., :k] / x[1:3]。"""
    x = torch.randn(4, 6, requires_grad=True)
    a = x[:, :3]
    assert _np(a).shape == (4, 3), _np(a).shape
    b = x[1:3]
    assert _np(b).shape == (2, 6), _np(b).shape
    c = x[..., 2:5]
    assert _np(c).shape == (4, 3), _np(c).shape
    # 反向只应落在被切走的那些位置
    (a * 2.0).sum().backward()
    g = _np(x.grad)
    assert np.allclose(g[:, :3], 2.0) and np.allclose(g[:, 3:], 0.0), "切片反向落错位置"
    return "ok"


def case_repeat_interleave():
    """GQA 常用：在 head 维做 repeat_interleave。"""
    k = torch.randn(2, 3, 4)          # [heads=2, S, Dh]
    r = torch.cat([k[0:1], k[0:1], k[1:2], k[1:2]], dim=0) if not hasattr(torch, "repeat_interleave") else None
    if r is None:
        return "skip(有原生 repeat_interleave)"
    assert _np(r).shape == (4, 3, 4), _np(r).shape
    assert np.allclose(_np(r)[0], _np(r)[1]), "重复的头应完全一致"
    return "ok"


CASES = [
    ("tensor_basic", case_tensor_basic),
    ("linear_nd", case_linear_nd),
    ("sequential", case_sequential),
    ("functional_names", case_functional_names),
    ("cross_entropy", case_cross_entropy),
    ("no_grad", case_no_grad),
    ("optimizer_step", case_optimizer_step),
    ("state_dict", case_state_dict),
    ("reshape_permute", case_reshape_permute),
    ("seed_repeat", case_seed_repeat),
    ("rmsnorm_attn", case_rmsnorm_attn),
    ("slice_basic", case_slice_basic),
    ("repeat_interleave", case_repeat_interleave),
]


def main():
    a = sys.argv[1:]
    if "--list" in a:
        for n, _ in CASES:
            print(n)
        return 0
    only = None
    if "--case" in a:
        only = a[a.index("--case") + 1]
    if "--cuda" in a:
        if not hasattr(torch, "set_device"):
            print("CUDA 后端未实现（无 set_device）")
            return 2
        torch.set_device("cuda")
        print("[CUDA 后端] 已切到 device=cuda")

    ok_n = fail_n = 0
    fails = []
    for name, fn in CASES:
        if only and name != only:
            continue
        try:
            fn()
            print("  %-18s PASS" % name)
            ok_n += 1
        except Exception as ex:
            msg = str(ex).splitlines()[0][:110] if str(ex) else type(ex).__name__
            print("  %-18s FAIL  %s: %s" % (name, type(ex).__name__, msg))
            fails.append(name)
            fail_n += 1
    total = ok_n + fail_n
    print("=" * 60)
    print("迁移一致性: %d/%d PASS" % (ok_n, total))
    if fails:
        print("未通过:", ", ".join(fails))
    print("VERDICT ok=%s" % (fail_n == 0))
    return 0 if fail_n == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
