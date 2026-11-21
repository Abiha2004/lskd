"""Unit tests:  python -m pytest -q tests   (or:  python tests/test_core.py)
ATTRIBUTION: IMPLEMENTED BY OUR TEAM."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lskd.data import augment, normalize  # noqa: E402
from lskd.losses import distillation_loss, total_loss, zscore  # noqa: E402
from lskd.models import build_model, count_parameters  # noqa: E402
from scripts.toy_example import run as toy  # noqa: E402


def test_zscore_properties():
    z = torch.randn(8, 100) * 7 + 3
    s = zscore(z)
    assert torch.allclose(s.mean(1), torch.zeros(8), atol=1e-5)          # zero mean
    assert torch.allclose(s.std(1), torch.ones(8), atol=1e-4)            # unit std
    assert torch.equal(s.argsort(1), z.argsort(1))                       # monotonic: ranking kept


def test_ls_ignores_scale_and_shift():
    """Core claim of the paper: a student whose logits are a*teacher+b (a>0) is PERFECT under LS
    but is penalised by vanilla KD."""
    t = torch.randn(16, 100) * 3
    s = 0.2 * t + 5.0
    assert distillation_loss(s, t, 2.0, standardize=True).item() < 1e-6
    assert distillation_loss(s, t, 4.0, standardize=False).item() > 1e-2


def test_mean_subtraction_alone_changes_nothing():
    """softmax is shift-invariant, so centring logits without dividing by std is a no-op."""
    t, s = torch.randn(4, 100), torch.randn(4, 100)
    a = distillation_loss(s, t, 4.0, standardize=False)
    b = distillation_loss(s - s.mean(1, keepdim=True), t - t.mean(1, keepdim=True), 4.0, standardize=False)
    assert torch.allclose(a, b, atol=1e-5)


def test_total_loss_shapes():
    s, t = torch.randn(4, 100, requires_grad=True), torch.randn(4, 100)
    y = torch.randint(0, 100, (4,))
    for m in ("ce", "kd", "kd_ls"):
        loss, ce, kd = total_loss(s, y, t, m, 2.0, 0.1, 9.0)
        loss.backward()
        assert torch.isfinite(loss)


def test_models():
    for arch, k, approx in (("resnet20", 100, 0.278e6), ("resnet56", 100, 0.861e6)):
        m = build_model(arch, k)
        assert m(torch.randn(2, 3, 32, 32)).shape == (2, k)
        assert abs(count_parameters(m) - approx) / approx < 0.01


def test_augment_and_normalize():
    x = torch.rand(5, 3, 32, 32)
    a = augment(x)
    assert a.shape == x.shape and a.min() >= 0 and a.max() <= 1
    assert normalize(a, "cifar100").shape == x.shape


def test_toy_example_matches_paper():
    r = toy(verbose=False)
    assert abs(r["S1"]["kl_raw"] - 0.1749) < 1e-3 and abs(r["S2"]["kl_raw"] - 0.3457) < 1e-3
    assert abs(r["S1"]["kl_z"] - 0.0995) < 1e-3 and r["S2"]["kl_z"] < 1e-9


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
