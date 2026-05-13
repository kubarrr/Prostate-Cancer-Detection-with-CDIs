"""Figures 1-3 reproductions."""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from . import analysis, config
from .tissue_sampling import TissuePool


# ── Fig. 1: histograms ────────────────────────────────────────────────────────

def plot_histograms(pools: Sequence[TissuePool],
                    *, bins: int = 50, save_path: Optional[Path] = None,
                    title: str = "Tissue distributions per modality") -> plt.Figure:
    n = len(pools)
    cols = 2 if n <= 4 else 3
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.5 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, pool in zip(axes, pools):
        for cls in ("healthy", "insPCa", "csPCa"):
            vals = np.asarray(getattr(pool, cls), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            ax.hist(vals, bins=bins, density=True, alpha=0.55,
                    color=config.TISSUE_COLORS[cls], label=cls,
                    edgecolor="white", linewidth=0.4)
        ax.set_title(pool.name)
        ax.set_ylabel("Density"); ax.set_xlabel("Signal intensity")
        ax.legend(frameon=False, fontsize=9)

    for ax in axes[len(pools):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig. 2: ROC grid ──────────────────────────────────────────────────────────

def plot_roc_grid(pools: Sequence[TissuePool],
                  tasks: Sequence[str] = tuple(analysis.ROC_TASKS.keys()),
                  *, save_path: Optional[Path] = None,
                  title: str = "ROC by delineation task") -> plt.Figure:
    cols = min(len(tasks), 3)
    rows = int(np.ceil(len(tasks) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(5.0 * cols, 4.4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for ax, task in zip(axes, tasks):
        ax.plot([0, 1], [0, 1], color="lightgray", lw=1, ls="--")
        for pool in pools:
            res = analysis.compute_roc(pool, task)
            color = config.MODALITY_COLORS.get(pool.name)
            ax.plot(res.fpr, res.tpr, color=color, lw=1.6,
                    label=f"{pool.name}: AUC={res.auc:.3f}")
        ax.set_title(task)
        ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1.01)
        ax.legend(loc="lower right", fontsize=8, frameon=False)

    for ax in axes[len(tasks):]:
        ax.axis("off")

    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── Fig. 3: overlays ──────────────────────────────────────────────────────────

def _pick_lesion_slice(prostate_mask: np.ndarray,
                       lesion_rois: Sequence[tuple[np.ndarray, bool]]) -> int:
    weights = np.array([m.sum() for m, _ in lesion_rois]) if lesion_rois else None
    if weights is not None and weights.size > 0 and weights.max() > 0:
        i = int(np.argmax(weights))
        zs = np.argwhere(lesion_rois[i][0]).T[0]
        return int(np.median(zs))
    zs = np.argwhere(prostate_mask > 0).T[0]
    return int(np.median(zs)) if len(zs) else prostate_mask.shape[0] // 2


def plot_patient_overlays(t2w: np.ndarray, adc: np.ndarray, cdis: np.ndarray,
                          prostate_mask: np.ndarray,
                          lesion_rois: Sequence[tuple[np.ndarray, bool]],
                          *, slice_z: Optional[int] = None,
                          save_path: Optional[Path] = None,
                          title: str = "Patient overlays") -> plt.Figure:
    if slice_z is None:
        slice_z = _pick_lesion_slice(prostate_mask, lesion_rois)
    z = max(0, min(slice_z, cdis.shape[0] - 1))

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [("T2w",       t2w, "gray",      None),
              ("ADC",       adc, "viridis_r", "ADC"),
              ("log(CDIs)", np.log(np.maximum(cdis, 1e-12)), "magma", "log(CDIs)")]

    for ax, (name, vol, cmap, cbar_label) in zip(axes, panels):
        if vol is None:
            ax.text(0.5, 0.5, f"{name}: not available", ha="center", va="center")
            ax.axis("off"); continue
        im = ax.imshow(vol[z], cmap=cmap)
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        if cbar_label:
            cbar.set_label(cbar_label)
        ax.contour(prostate_mask[z] > 0, levels=[0.5], colors="white", linewidths=0.7)
        for m, is_cs in lesion_rois:
            ax.contour(m[z] > 0, levels=[0.5],
                       colors=[config.TISSUE_COLORS["csPCa" if is_cs else "insPCa"]],
                       linewidths=1.4)
        ax.set_title(name)
        ax.set_xticks([]); ax.set_yticks([])

    legend_elements = [
        mlines.Line2D([], [], color="white", linewidth=1.5, label="Prostate gland"),
        mlines.Line2D([], [], color=config.TISSUE_COLORS["csPCa"],  linewidth=2, label="csPCa"),
        mlines.Line2D([], [], color=config.TISSUE_COLORS["insPCa"], linewidth=2, label="insPCa"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle(f"{title} (slice z={z})")
    fig.tight_layout(rect=(0, 0.08, 1, 0.96))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_patient_grid(patients: Sequence[dict],
                      *, save_path: Optional[Path] = None,
                      title: str = "Patient overlays — T2w / ADC / log(CDIs)",
                      cmap_cdis: str = "magma") -> plt.Figure:
    """Each entry: {pid, t2w, adc, cdis, prostate_mask, lesion_rois, [slice_z], [tag]}."""
    n = len(patients)
    fig, axes = plt.subplots(n, 3, figsize=(16, 4.5 * n), squeeze=False)

    for row, p in enumerate(patients):
        z = p.get("slice_z") or _pick_lesion_slice(p["prostate_mask"], p["lesion_rois"])
        z = int(max(0, min(z, p["cdis"].shape[0] - 1)))
        panels = [
            ("T2w",       p["t2w"][z]  if p["t2w"]  is not None else None, "gray",      None),
            ("ADC",       p["adc"][z]  if p["adc"]  is not None else None, "viridis_r", "ADC"),
            ("log(CDIs)", np.log(np.maximum(p["cdis"][z], 1e-12)),         cmap_cdis,   "log(CDIs)"),
        ]
        for col, (name, img2d, cm, cbar_label) in enumerate(panels):
            ax = axes[row, col]
            if img2d is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center"); ax.axis("off"); continue
            im = ax.imshow(img2d, cmap=cm)
            cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if cbar_label:
                cbar.set_label(cbar_label)
            ax.contour(p["prostate_mask"][z] > 0, levels=[0.5],
                       colors="white", linewidths=0.7)
            for m, is_cs in p["lesion_rois"]:
                ax.contour(m[z] > 0, levels=[0.5],
                           colors=[config.TISSUE_COLORS["csPCa" if is_cs else "insPCa"]],
                           linewidths=1.4)
            if row == 0:
                ax.set_title(name)
            ax.set_xticks([]); ax.set_yticks([])
        axes[row, 0].set_ylabel(f"{p['pid']}\n{p.get('tag', '')}", fontsize=10)

    legend_elements = [
        mlines.Line2D([], [], color="white", linewidth=1.5, label="Prostate gland"),
        mlines.Line2D([], [], color=config.TISSUE_COLORS["csPCa"],  linewidth=2, label="csPCa"),
        mlines.Line2D([], [], color=config.TISSUE_COLORS["insPCa"], linewidth=2, label="insPCa"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, 0.01))
    fig.suptitle(title)
    fig.tight_layout(rect=(0, max(0.03, 0.1 / max(n, 1)), 1, 0.97))
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


# ── AUC table ─────────────────────────────────────────────────────────────────

def auc_table(pools: Sequence[TissuePool],
              tasks: Sequence[str] = tuple(analysis.ROC_TASKS.keys())):
    import pandas as pd
    rows = []
    for task in tasks:
        for pool in pools:
            r = analysis.compute_roc(pool, task)
            rows.append({"task": task, "modality": pool.name,
                         "auc": r.auc, "n_pos": r.n_pos, "n_neg": r.n_neg,
                         "se": analysis.auc_se(r.auc, r.n_pos, r.n_neg)})
    return pd.DataFrame(rows)
