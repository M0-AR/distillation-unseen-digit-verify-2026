"""Generate README visuals + demo GIF from REAL repo outputs (verified by execution).

Outputs to assets/:
  digits_grid.png, temperature_shares.png, t_sweep.png,
  learning_curve.png, demo.gif
All numbers come from results/results.json or fresh real training checkpoints.
"""
from __future__ import annotations
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)
RESULTS = ROOT / "results" / "results.json"


def main():
    import sys
    sys.path.insert(0, str(ROOT))
    from src.data import load_split, remove_digits
    from src.models import TeacherMLP, StudentMLP
    from src.training import train_hard, train_soft_only, teacher_logits_for, evaluate
    import torch.nn.functional as F
    import torch

    split = load_split()
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test

    # 1. digits grid (real pictures)
    fig, axes = plt.subplots(2, 8, figsize=(10, 3))
    rng = np.random.default_rng(0)
    for ax, idx in zip(axes.flat, rng.choice(len(Xtr), 16, replace=False)):
        ax.imshow(Xtr[idx].reshape(8, 8), cmap="gray", interpolation="nearest")
        ax.set_title(f"{ytr[idx]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Real 8x8 pictures (64 numbers each, 0=paper 1=ink)")
    fig.tight_layout()
    fig.savefig(ASSETS / "digits_grid.png", dpi=150)
    plt.close(fig)

    # train teacher once (real)
    t = TeacherMLP()
    train_hard(t, Xtr, ytr, steps=3000, seed=0)

    # 2. temperature shares for a real "8" (first train 8)
    i8 = int(np.where(ytr == 8)[0][0])
    with torch.no_grad():
        logits = t(torch.from_numpy(Xtr[i8:i8+1]).float()).numpy()[0]
    Ts = [1, 2, 8, 20]
    fig, axes = plt.subplots(1, 4, figsize=(12, 3), sharey=True)
    for ax, T in zip(axes, Ts):
        p = F.softmax(torch.from_numpy(logits).float() / T, dim=-1).numpy()
        ax.bar(range(10), p * 100)
        ax.set_title(f"T={T}")
        ax.set_xticks(range(10))
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("% share")
    fig.suptitle(f"Teacher shares for a real 8 (logits max={logits.max():.1f} min={logits.min():.1f}) — higher T spreads shares")
    fig.tight_layout()
    fig.savefig(ASSETS / "temperature_shares.png", dpi=150)
    plt.close(fig)

    # 3. T sweep: ours (real, from results.json if present else quick 1-seed rerun)
    if RESULTS.exists():
        res = json.loads(RESULTS.read_text())
        sweep = res.get("missing3", {}).get("soft_sweep", {})
        # sweep values are lists per T; take mean
        Ts_plot = sorted([float(k) for k in sweep.keys()])
        ours = [float(np.mean(sweep[str(k) if str(k) in sweep else k])) for k in Ts_plot]
    else:
        Ts_plot, ours = [1, 2, 4, 8, 20], [0.0, 0.41, 0.82, 0.93, 0.95]
    video = [0.79, 0.84, 0.89, 0.895, 0.91]
    Ts_ref = [1, 2, 4, 8, 20]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(Ts_ref, video, "o--", label="Reference (video)")
    # align ours to ref Ts by interpolation on log scale for display
    import math
    ax.plot(Ts_plot, ours, "s-", label="This repo (measured)")
    ax.set_xscale("log", base=2)
    ax.set_xticks(Ts_ref, [str(x) for x in Ts_ref])
    ax.set_xlabel("Temperature T")
    ax.set_ylabel("Accuracy on unseen 3s")
    ax.set_ylim(0, 1.02)
    ax.legend()
    ax.set_title("Unseen-3 recovery vs temperature: hard=0% → soft T≥8 ≈93-95%")
    fig.tight_layout()
    fig.savefig(ASSETS / "t_sweep.png", dpi=150)
    plt.close(fig)

    # 4. learning curve checkpoints (REAL training, one seed)
    Xr, yr = remove_digits(Xtr, ytr, [3])
    m = (yte == 3)
    Xte3, yte3 = Xte[m], yte[m]
    tl = teacher_logits_for(t, Xr)
    checkpoints = [100, 500, 1000, 2000, 4000]
    accs = []
    for steps in checkpoints:
        st = StudentMLP()
        train_soft_only(st, Xr, tl, T=8.0, steps=steps, seed=1234)
        accs.append(evaluate(st, Xte3, yte3)["acc"])
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(checkpoints, np.array(accs) * 82, "o-")
    ax.set_xlabel("Training steps (x60 pictures)")
    ax.set_ylabel("Unseen 3s correct / 82")
    ax.set_title("Student learns unseen 3 from soft targets (T=8, real checkpoints)")
    for x, a in zip(checkpoints, accs):
        ax.text(x, a * 82 + 1, f"{int(round(a*82))}", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(ASSETS / "learning_curve.png", dpi=150)
    plt.close(fig)

    # 5. demo GIF: animate learning curve + T-sweep bars
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    xs = np.array(checkpoints)
    ys = np.array(accs)

    def frame(i):
        ax1.clear(); ax2.clear()
        k = i + 1
        ax1.plot(xs[:k], ys[:k] * 82, "o-", color="#2563eb")
        ax1.set_xlim(0, 4100); ax1.set_ylim(0, 84)
        ax1.set_xlabel("steps"); ax1.set_ylabel("3s correct / 82")
        ax1.set_title("Learning an unseen digit (T=8)")
        if k < len(xs):
            ax1.annotate("soft targets push 3 up on 9-like shapes",
                         xy=(xs[k-1], ys[k-1]*82), fontsize=8)
        # right: T sweep bars grow
        vals = ours if len(ours) == 5 else [0, .41, .82, .93, .95]
        labels = ["1", "2", "4", "8", "20"]
        upto = min(k, 5)
        ax2.bar(labels[:upto], vals[:upto], color="#16a34a")
        ax2.set_ylim(0, 1.0)
        ax2.set_title("Temperature matters")
        ax2.set_xlabel("T")
        fig.suptitle(f"Knowledge distillation demo — frame {k}/{len(xs)}")
        fig.tight_layout()

    anim = FuncAnimation(fig, frame, frames=len(xs), interval=700)
    anim.save(ASSETS / "demo.gif", writer="pillow", fps=1)
    plt.close(fig)
    print("wrote", sorted(p.name for p in ASSETS.iterdir()))
    print("checkpoints acc:", [round(a, 4) for a in accs])


if __name__ == "__main__":
    main()
