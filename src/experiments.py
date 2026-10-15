"""Full benchmark suite reproducing every number in the video.

Run: python -m src.experiments --mode fast   (CI, ~2-4 min CPU)
     python -m src.experiments --mode full   (paper, ~30-60 min CPU)

Outputs results/results.json + results/SUMMARY.md
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

from src.data import load_split, remove_digits, make_mixed_pictures, make_noise_pictures
from src.models import TeacherMLP, StudentMLP, param_count
from src.training import train_hard, train_soft_only, teacher_logits_for, evaluate
from src.utils import set_seed

RESULTS = Path(__file__).resolve().parent.parent / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def run_teacher(Xtr, ytr, Xte, yte, seed=0, steps=3000):
    set_seed(seed)
    t = TeacherMLP()
    train_hard(t, Xtr, ytr, steps=steps, seed=seed)
    ev = evaluate(t, Xte, yte)
    return t, ev


def run_student_hard(Xtr, ytr, Xte, yte, seed, steps=4000):
    set_seed(1000 + seed)
    s = StudentMLP()
    train_hard(s, Xtr, ytr, steps=steps, seed=1000 + seed)
    return s, evaluate(s, Xte, yte)


def run_student_soft(Xtr, Xte, yte, teacher_logits, T, seed, steps=4000):
    set_seed(2000 + seed)
    s = StudentMLP()
    train_soft_only(s, Xtr, teacher_logits, T=T, steps=steps, seed=2000 + seed)
    return s, evaluate(s, Xte, yte)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fast", "full"], default="fast")
    args = ap.parse_args()

    if args.mode == "fast":
        teacher_steps, student_steps = 3000, 4000
        n_rep, T_list = 2, [1.0, 8.0]
        n_mixed, leave_digits = 2000, [3]
    else:
        teacher_steps, student_steps = 3000, 4000
        n_rep, T_list = 5, [1.0, 2.0, 4.0, 8.0, 20.0]
        n_mixed, leave_digits = 10000, [3]

    out: dict = {"mode": args.mode, "started": time.strftime("%Y-%m-%dT%H:%M:%S")}
    split = load_split()
    Xtr, ytr, Xte, yte = split.X_train, split.y_train, split.X_test, split.y_test
    out["split"] = {"n_train": len(ytr), "n_test": len(yte),
                    "n_train_is_3": int((ytr == 3).sum()),
                    "n_test_is_3": int((yte == 3).sum())}
    out["params"] = {"teacher": param_count(TeacherMLP()),
                     "student": param_count(StudentMLP())}

    # 1. teacher baseline
    t, tev = run_teacher(Xtr, ytr, Xte, yte, seed=0, steps=teacher_steps)
    out["teacher"] = {k: v for k, v in tev.items() if k != "preds"}
    print(f"teacher acc={tev['acc']:.4f} ({tev['correct']}/{tev['n']})")
    full_teacher_logits = teacher_logits_for(t, Xtr)

    # 2. full-data students hard vs soft
    hard_accs, soft_accs = [], []
    for s in range(n_rep):
        _, ev = run_student_hard(Xtr, ytr, Xte, yte, seed=s, steps=student_steps)
        hard_accs.append(ev["acc"])
        _, ev2 = run_student_soft(Xtr, Xte, yte, full_teacher_logits, T=8.0,
                                  seed=s, steps=student_steps)
        soft_accs.append(ev2["acc"])
    out["full_data"] = {"hard_mean": float(np.mean(hard_accs)),
                        "soft_T8_mean": float(np.mean(soft_accs)),
                        "hard": hard_accs, "soft": soft_accs}
    print(f"full hard={np.mean(hard_accs):.4f} soft={np.mean(soft_accs):.4f}")

    # 3. missing-3: hard vs soft at T sweep
    Xtr_no3, ytr_no3 = remove_digits(Xtr, ytr, [3])
    m3 = (yte == 3)
    Xte3, yte3 = Xte[m3], yte[m3]
    out["missing3"] = {"n_train_no3": len(ytr_no3), "n_test_3": int(m3.sum())}
    # teacher logits for the 899 (recompute with teacher trained on FULL data,
    # exactly as video: bigger network had seen threes but gives shares only
    # for these same non-three pictures)
    tlog_no3 = teacher_logits_for(t, Xtr_no3)
    hard3, soft3 = [], []
    for s in range(n_rep):
        _, ev = run_student_hard(Xtr_no3, ytr_no3, Xte3, yte3, seed=10 + s,
                                 steps=student_steps)
        hard3.append(ev["acc"])
    out["missing3"]["hard_per_seed"] = hard3
    print(f"missing-3 hard: {hard3} (expect all 0.0)")
    sweep = {}
    for T in T_list:
        accs = []
        for s in range(n_rep):
            _, ev = run_student_soft(Xtr_no3, Xte3, yte3, tlog_no3, T=T,
                                     seed=100 + int(T * 10) + s,
                                     steps=student_steps)
            accs.append(ev["acc"])
        sweep[str(T)] = accs
        print(f"missing-3 soft T={T}: mean={np.mean(accs):.4f} {accs}")
    out["missing3"]["soft_sweep"] = sweep

    # 4. mixed + noise transfer (video: mixed 91%, noise 55%)
    for name, Xm in [("mixed", make_mixed_pictures(Xtr, n_mixed, seed=0)),
                     ("noise", make_noise_pictures(n_mixed, seed=1))]:
        tlog_m = teacher_logits_for(t, Xm)
        accs = []
        for s in range(min(n_rep, 2)):
            _, ev = run_student_soft(Xm, Xte, yte, tlog_m, T=8.0,
                                     seed=500 + s, steps=min(student_steps, 4000))
            accs.append(ev["acc"])
        out[name] = {"mean": float(np.mean(accs)), "seeds": accs}
        print(f"{name}: {accs}")

    # 5. mistake-copy: teacher wrong on 17? count overlap
    import torch
    t.eval()
    with torch.no_grad():
        import torch as _t
        preds_t = []
        for i in range(0, len(Xte), 512):
            xb = _t.from_numpy(Xte[i:i + 512]).float()
            preds_t.append(t(xb).argmax(-1).numpy())
    import numpy as _np
    preds_t = _np.concatenate(preds_t)
    wrong_mask = (preds_t != yte)
    out["teacher_wrong_n"] = int(wrong_mask.sum())
    # one soft student vs one hard student overlap
    _, evh = run_student_hard(Xtr, ytr, Xte, yte, seed=0, steps=student_steps)
    sh, _ = run_student_soft(Xtr, Xte, yte, full_teacher_logits, T=8.0, seed=0,
                             steps=student_steps)
    with torch.no_grad():
        ps, ph = [], []
        for i in range(0, len(Xte), 512):
            xb = _t.from_numpy(Xte[i:i + 512]).float()
            ps.append(sh(xb).argmax(-1).numpy())
    import src.training as _tr
    # hard student preds from evaluate
    evh_full = _tr.evaluate(StudentMLP(), Xte[:1], yte[:1])  # placeholder unused
    # recompute hard preds properly
    set_seed(1000)
    sh_hard = StudentMLP()
    from src.training import train_hard as _th
    _th(sh_hard, Xtr, ytr, steps=student_steps, seed=1000)
    with torch.no_grad():
        ph = []
        for i in range(0, len(Xte), 512):
            xb = _t.from_numpy(Xte[i:i + 512]).float()
            ph.append(sh_hard(xb).argmax(-1).numpy())
    ph = _np.concatenate(ph)
    ps = _np.concatenate(ps)
    teacher_wrong_idx = _np.where(wrong_mask)[0]
    out["mistake_copy"] = {
        "soft_same_wrong_as_teacher": int(((ps[teacher_wrong_idx] == preds_t[teacher_wrong_idx])).sum()),
        "hard_same_wrong_as_teacher": int(((ph[teacher_wrong_idx] == preds_t[teacher_wrong_idx])).sum()),
    }
    print("mistake copy:", out["mistake_copy"])

    (RESULTS / "results.json").write_text(json.dumps(out, indent=2))
    with open(RESULTS / "SUMMARY.md", "w") as f:
        f.write("# Distillation unseen-digit benchmark — summary\n\n")
        f.write(f"mode={args.mode}\n\n")
        f.write(f"teacher: {out['teacher']['correct']}/{out['teacher']['n']} "
                f"={out['teacher']['acc']:.4f}\n\n")
        f.write(f"full hard mean={out['full_data']['hard_mean']:.4f} "
                f"soft mean={out['full_data']['soft_T8_mean']:.4f}\n\n")
        f.write(f"missing-3 hard={out['missing3']['hard_per_seed']} "
                f"sweep={out['missing3']['soft_sweep']}\n\n")
        f.write(f"mixed={out['mixed']} noise={out['noise']}\n\n")
        f.write(f"mistake_copy={out['mistake_copy']}\n")
    print("wrote results/results.json")


if __name__ == "__main__":
    main()
