import sys
sys.path.insert(0, "/home/md/src/distillation-unseen-digit-verify-2026")

import numpy as np
import torch


def test_teacher_params():
    from src.models import TeacherMLP, param_count
    assert param_count(TeacherMLP()) == 190410


def test_student_params():
    from src.models import StudentMLP, param_count
    assert param_count(StudentMLP()) == 9610


def test_dataset_shape():
    from sklearn.datasets import load_digits
    X, y = load_digits(return_X_y=True)
    assert X.shape == (1797, 64)
    assert set(np.unique(y)) == set(range(10))


def test_split_video_numbers():
    from src.data import load_split
    s = load_split()
    assert len(s.y_train) == 1000 and len(s.y_test) == 797
    assert int((s.y_train == 3).sum()) == 101
    assert int((s.y_test == 3).sum()) == 82


def test_remove_digits():
    from src.data import load_split, remove_digits
    s = load_split()
    Xr, yr = remove_digits(s.X_train, s.y_train, [3])
    assert len(yr) == 899 and (yr != 3).all()


def test_temperature_softmax_order_preserved():
    from src.utils import softmax_with_temperature
    logits = torch.tensor([[8.4, 0.4, 0.4, -8.3]])
    for T in [1, 2, 8, 20]:
        p = softmax_with_temperature(logits, T)
        assert abs(p.sum().item() - 1.0) < 1e-5
        assert p[0, 0] > p[0, 1]  # order preserved


def test_temperature_spreads_shares():
    from src.utils import softmax_with_temperature
    logits = torch.tensor([[8.4, 0.4]])
    p1 = softmax_with_temperature(logits, 1.0)
    p8 = softmax_with_temperature(logits, 8.0)
    # higher T -> more even (entropy up)
    e1 = -(p1 * p1.log()).sum().item()
    e8 = -(p8 * p8.log()).sum().item()
    assert e8 > e1


def test_soft_training_runs():
    from src.data import load_split, remove_digits
    from src.models import TeacherMLP, StudentMLP
    from src.training import train_hard, train_soft_only, teacher_logits_for, evaluate
    s = load_split()
    t = TeacherMLP()
    train_hard(t, s.X_train, s.y_train, steps=20, seed=0)
    tl = teacher_logits_for(t, s.X_train[:120])
    st = StudentMLP()
    train_soft_only(st, s.X_train[:120], tl, T=8.0, steps=20, seed=0)
    ev = evaluate(st, s.X_test[:50], s.y_test[:50])
    assert 0.0 <= ev["acc"] <= 1.0


def test_hard_missing3_is_zero():
    from src.data import load_split, remove_digits
    from src.models import StudentMLP
    from src.training import train_hard, evaluate
    s = load_split()
    Xr, yr = remove_digits(s.X_train, s.y_train, [3])
    m = (s.y_test == 3)
    h = StudentMLP()
    train_hard(h, Xr, yr, steps=400, seed=10)
    ev = evaluate(h, s.X_test[m], s.y_test[m])
    assert ev["acc"] == 0.0


def test_mixed_and_noise_shapes():
    from src.data import load_split, make_mixed_pictures, make_noise_pictures
    s = load_split()
    assert make_mixed_pictures(s.X_train, 16, seed=0).shape == (16, 64)
    assert make_noise_pictures(16, seed=0).shape == (16, 64)


def test_live_uci_reachable():
    from src.live_verify import verify_live
    try:
        r = verify_live()
    except Exception:
        return  # offline CI: do not fail
    assert r["sklearn_shape"] == [1797, 64]
