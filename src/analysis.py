"""Histograms, ROC, AUC."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve

from .tissue_sampling import TissuePool


ROC_TASKS = {
    "csPCa vs healthy":  (("csPCa",),          ("healthy",)),
    "insPCa vs healthy": (("insPCa",),         ("healthy",)),
    "csPCa vs insPCa":   (("csPCa",),          ("insPCa",)),
    "csPCa vs other":    (("csPCa",),          ("healthy", "insPCa")),
    "PCa vs healthy":    (("csPCa", "insPCa"), ("healthy",)),
}


@dataclass
class ROCResult:
    task: str
    modality: str
    fpr: np.ndarray
    tpr: np.ndarray
    auc: float
    n_pos: int
    n_neg: int
    higher_is_positive: bool


def compute_roc(pool: TissuePool, task: str) -> ROCResult:
    pos_labels, neg_labels = ROC_TASKS[task]
    pos = np.concatenate([np.asarray(getattr(pool, k), dtype=float) for k in pos_labels])
    neg = np.concatenate([np.asarray(getattr(pool, k), dtype=float) for k in neg_labels])
    pos = pos[np.isfinite(pos)]
    neg = neg[np.isfinite(neg)]

    if len(pos) == 0 or len(neg) == 0:
        return ROCResult(task, pool.name, np.array([0.0, 1.0]), np.array([0.0, 1.0]),
                         float("nan"), len(pos), len(neg), True)

    y_true  = np.concatenate([np.zeros(len(neg)), np.ones(len(pos))])
    y_score = np.concatenate([neg, pos])
    auc = roc_auc_score(y_true, y_score)
    higher = True
    if auc < 0.5:
        # Lower-is-positive (e.g. ADC for cancer): flip sign.
        y_score = -y_score
        auc = 1.0 - auc
        higher = False
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return ROCResult(task, pool.name, fpr, tpr, float(auc),
                     int(len(pos)), int(len(neg)), higher)


def auc_se(auc: float, n_pos: int, n_neg: int) -> float:
    """Hanley-McNeil SE."""
    A = float(auc); nP = max(int(n_pos), 1); nN = max(int(n_neg), 1)
    Q1 = A / (2 - A) - A * A
    Q2 = 2 * A * A / (1 + A) - A * A
    var = (A * (1 - A) + (nP - 1) * Q1 + (nN - 1) * Q2) / (nP * nN)
    return float(np.sqrt(max(var, 0.0)))
