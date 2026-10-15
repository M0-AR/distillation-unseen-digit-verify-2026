"""Data loading for UCI / sklearn digits (1797 x 8x8, values 0..16).

Video mapping: 1000 learning + 797 test = 1797 total.
We reproduce the video EXACTLY with train_test_split(test_size=797,
random_state=9) WITHOUT stratify, which yields 101 threes in train and
82 threes in test (899 stay when threes removed). Verified by execution.

Features are scaled to [0,1] by /16 (0=empty paper, 1=full ink).
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split


@dataclass
class Split:
    X_train: np.ndarray
    y_train: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray


def load_split(test_size: int = 797, random_state: int = 9, stratify_opt: bool = False) -> Split:
    X, y = load_digits(return_X_y=True)
    X = X.astype(np.float32) / 16.0
    y = y.astype(np.int64)
    strat = y if stratify_opt else None
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat
    )
    return Split(Xtr, ytr, Xte, yte)


def remove_digits(X: np.ndarray, y: np.ndarray, digits: list[int]):
    mask = np.ones(len(y), dtype=bool)
    for d in digits:
        mask &= (y != d)
    return X[mask], y[mask]


def make_mixed_pictures(X_train: np.ndarray, n: int, seed: int = 0) -> np.ndarray:
    """Each square copied from same position of a random real picture.

    Result is not a digit; no human could label it, but teacher still
    gives 10 shares. Video: 10k mixed -> student reaches 91% on real test.
    """
    rng = np.random.default_rng(seed)
    n_train, dim = X_train.shape
    idx = rng.integers(0, n_train, size=(n, dim))
    rows = np.arange(dim)[None, :].repeat(n, axis=0)
    # fancy: for each output (i,j), take X_train[idx[i,j], j]
    out = X_train[idx, rows]
    return out.astype(np.float32)


def make_noise_pictures(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.random((n, 64), dtype=np.float32)
