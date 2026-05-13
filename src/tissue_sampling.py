"""Resample masks/images to the DWI grid and pool per-tissue voxel values."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import SimpleITK as sitk

from . import config


# ── resampling ────────────────────────────────────────────────────────────────

def resample_to_reference(image: sitk.Image, ref: sitk.Image,
                          *, interpolator=sitk.sitkLinear,
                          default: float = 0.0) -> sitk.Image:
    r = sitk.ResampleImageFilter()
    r.SetReferenceImage(ref)
    r.SetInterpolator(interpolator)
    r.SetDefaultPixelValue(default)
    return r.Execute(image)


def to_array(img: sitk.Image) -> np.ndarray:
    return sitk.GetArrayFromImage(img).astype(np.float32)


def voxel_spacing_xyz(img: sitk.Image) -> tuple[float, float, float]:
    sx, sy, sz = img.GetSpacing()
    return float(sx), float(sy), float(sz)


def resample_mask(mask_img: sitk.Image, ref: sitk.Image) -> np.ndarray:
    """Mask -> bool array on ref grid (nearest-neighbor)."""
    arr = to_array(resample_to_reference(mask_img, ref, interpolator=sitk.sitkNearestNeighbor))
    return arr > 0


# ── per-patient bundle ────────────────────────────────────────────────────────

@dataclass
class PatientBundle:
    pid: str
    voxel_spacing: tuple[float, float, float]
    signals: dict[int, np.ndarray]
    t2w: Optional[np.ndarray]
    adc: Optional[np.ndarray]
    prostate_mask: np.ndarray
    pz_mask: Optional[np.ndarray]
    tz_mask: Optional[np.ndarray]
    lesion_rois: list[tuple[np.ndarray, bool]] = field(default_factory=list)
    ref: Optional[sitk.Image] = None


def assign_zone(roi: np.ndarray, pz: Optional[np.ndarray], tz: Optional[np.ndarray]
                ) -> str:
    """Return 'PZ', 'TZ', 'AS' (anterior stroma / unknown), based on max overlap."""
    if pz is None and tz is None:
        return "?"
    npz = int(np.sum(roi & pz)) if pz is not None else 0
    ntz = int(np.sum(roi & tz)) if tz is not None else 0
    if npz == 0 and ntz == 0:
        return "AS"
    return "PZ" if npz >= ntz else "TZ"


# ── tissue value pooling ──────────────────────────────────────────────────────

@dataclass
class TissuePool:
    name: str
    healthy: np.ndarray
    insPCa: np.ndarray
    csPCa: np.ndarray

    def all_pos_neg(self, positive: Sequence[str], negative: Sequence[str]
                    ) -> tuple[np.ndarray, np.ndarray]:
        pos = np.concatenate([getattr(self, k) for k in positive]) if positive else np.array([])
        neg = np.concatenate([getattr(self, k) for k in negative]) if negative else np.array([])
        return pos, neg


def pool_modality(name: str, per_patient_values: list[dict]) -> TissuePool:
    def _stack(key: str) -> np.ndarray:
        arrs = [np.asarray(v[key], dtype=float).ravel() for v in per_patient_values]
        return np.concatenate(arrs) if arrs else np.array([], dtype=float)
    return TissuePool(name, _stack("healthy"), _stack("insPCa"), _stack("csPCa"))


def sample_modality(volume: np.ndarray, healthy_idx: np.ndarray,
                    lesion_rois: list[tuple[np.ndarray, bool]],
                    *, per_voxel_lesions: bool = True) -> dict:
    """Pull (healthy, insPCa, csPCa) value arrays from one volume."""
    healthy_vals = volume[healthy_idx[:, 0], healthy_idx[:, 1], healthy_idx[:, 2]]
    cs: list[float] = []
    ins: list[float] = []
    for m, is_cs in lesion_rois:
        if not m.any():
            continue
        vals = volume[m].tolist() if per_voxel_lesions else [float(np.median(volume[m]))]
        (cs if is_cs else ins).extend(vals)
    return {"healthy": np.asarray(healthy_vals, dtype=float),
            "insPCa":  np.asarray(ins,          dtype=float),
            "csPCa":   np.asarray(cs,           dtype=float)}


def pick_healthy_indices(prostate_mask: np.ndarray,
                         lesion_rois: list[tuple[np.ndarray, bool]],
                         *, max_voxels: int = config.HEALTHY_VOXELS_PER_PATIENT,
                         seed: int = config.RNG_SEED) -> np.ndarray:
    pm = prostate_mask > 0
    occ = np.zeros_like(pm)
    for m, _ in lesion_rois:
        occ |= (m > 0)
    idx = np.argwhere(pm & ~occ)
    if len(idx) > max_voxels:
        rng = np.random.default_rng(seed)
        idx = idx[rng.choice(len(idx), max_voxels, replace=False)]
    return idx
