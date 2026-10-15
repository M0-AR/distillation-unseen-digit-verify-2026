"""Training loops faithful to the video + Hinton et al. 2015.

Hard (right answers): CrossEntropy at T=1.
Soft (teacher shares): KL(teacher_T || student_T) * T^2, no hard labels,
  exactly the 'new way uses no right answers at all' from the video.
Paper section 2.1: gradients scale as 1/T^2 so multiply by T^2.

One step = 60 pictures. Teacher 3000 steps, student 4000 steps.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader


def train_hard(model, X: np.ndarray, y: np.ndarray, steps: int,
               batch: int = 60, lr: float = 1e-3, seed: int = 0,
               log_every: int = 0) -> dict:
    g = torch.Generator().manual_seed(seed)
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()
    idx = torch.randperm(len(ds), generator=g).tolist()
    ptr = 0
    for step in range(1, steps + 1):
        if ptr + batch > len(idx):
            idx = torch.randperm(len(ds), generator=g).tolist()
            ptr = 0
        b = idx[ptr:ptr + batch]
        ptr += batch
        xb = torch.from_numpy(X[np.array(b)]).float()
        yb = torch.from_numpy(y[np.array(b)]).long()
        opt.zero_grad()
        logits = model(xb)
        loss = F.cross_entropy(logits, yb)
        loss.backward()
        opt.step()
    return {"steps": steps}


@torch.no_grad()
def teacher_logits_for(model, X: np.ndarray, batch: int = 512) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float()
        out.append(model(xb).cpu().numpy())
    return np.concatenate(out, axis=0)


def train_soft_only(student, X: np.ndarray, teacher_logits: np.ndarray,
                    T: float, steps: int, batch: int = 60,
                    lr: float = 1e-3, seed: int = 0) -> dict:
    """Pure distillation: student never sees hard labels."""
    assert len(X) == len(teacher_logits)
    g = torch.Generator().manual_seed(seed)
    Tl = torch.from_numpy(teacher_logits).float()
    opt = torch.optim.Adam(student.parameters(), lr=lr)
    student.train()
    idx = torch.randperm(len(X), generator=g).tolist()
    ptr = 0
    for _ in range(steps):
        if ptr + batch > len(idx):
            idx = torch.randperm(len(X), generator=g).tolist()
            ptr = 0
        b = np.array(idx[ptr:ptr + batch])
        ptr += batch
        xb = torch.from_numpy(X[b]).float()
        tb = Tl[b]
        s = student(xb)
        # KLDiv: input log-probs student/T, target probs teacher/T
        loss = F.kl_div(
            F.log_softmax(s / T, dim=-1),
            F.softmax(tb / T, dim=-1),
            reduction="batchmean",
        ) * (T * T)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return {"steps": steps, "T": T}


@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, batch: int = 512) -> dict:
    import collections
    model.eval()
    preds = []
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).float()
        preds.append(model(xb).argmax(-1).cpu().numpy())
    preds = np.concatenate(preds)
    y = np.asarray(y)
    acc = float((preds == y).mean())
    per_class = {}
    for d in range(10):
        m = (y == d)
        if m.sum():
            per_class[int(d)] = float((preds[m] == d).mean())
    return {"acc": acc, "n": len(y), "correct": int((preds == y).sum()),
            "per_class": per_class, "preds": preds}
