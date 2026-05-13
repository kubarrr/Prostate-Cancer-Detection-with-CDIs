"""Synthetic Correlated Diffusion Imaging — Wong et al. 2022, Eq. 3-5."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.ndimage import gaussian_filter

from . import config


def estimate_adc(signals: Mapping[int, np.ndarray]) -> np.ndarray:
    """Per-voxel LSQ ADC from multi-b DWI (Eq. 5)."""
    bs = np.asarray(sorted(signals.keys()), dtype=np.float64)
    log_s = np.stack([np.log(np.maximum(signals[int(b)], 1.0)) for b in bs], axis=0)
    db = bs - bs.mean()
    num = np.einsum("i,i...->...", db, log_s - log_s.mean(axis=0))
    den = float(np.sum(db ** 2))
    return (-num / max(den, 1e-12)).astype(np.float32)


def synthesize(s_ref: np.ndarray, adc: np.ndarray,
               target_bvalues: Iterable[int], b_ref: int = config.B_REF,
               ) -> dict[int, np.ndarray]:
    """Synthesise S_b from S_{b_ref} and ADC (Eq. 4)."""
    out: dict[int, np.ndarray] = {}
    for b in target_bvalues:
        s = s_ref * np.exp(-(b - b_ref) * adc)
        out[int(b)] = np.maximum(s, 1e-6).astype(np.float32)
    return out


@dataclass
class CDIsResult:
    cdis: np.ndarray
    log_cdis: np.ndarray
    adc: np.ndarray
    z_calibration: float
    rho: dict[int, float]


def _sigma_voxels(spacing_xyz: Sequence[float]) -> tuple[float, float, float]:
    dx, dy, dz = spacing_xyz
    sx = config.SIGMA_INPLANE_MM / max(dx, 1e-6)
    sy = config.SIGMA_INPLANE_MM / max(dy, 1e-6)
    sz = config.SIGMA_THROUGH_MM / max(dz, 1e-6)
    return (sz, sy, sx)


def compute_cdis(signals_native: Mapping[int, np.ndarray],
                 prostate_mask: np.ndarray,
                 voxel_spacing_xyz: Sequence[float],
                 rho: Optional[Mapping[int, float]] = None,
                 b_synth: Iterable[int] = config.B_SYNTH,
                 b_ref: int = config.B_REF,
                 ) -> CDIsResult:
    adc = estimate_adc(signals_native)
    s_ref = signals_native[b_ref]
    mixed: dict[int, np.ndarray] = {b_ref: s_ref, **synthesize(s_ref, adc, b_synth, b_ref=b_ref)}

    if rho is None:
        rho = {b: 1.0 for b in mixed}
    else:
        rho = {int(b): float(rho.get(int(b), 1.0)) for b in mixed}

    # Σ_b ρ_b log S_b  -> Gaussian filter  -> exp  ≡  weighted geometric mean.
    weighted_log = np.zeros_like(s_ref, dtype=np.float64)
    for b, s in mixed.items():
        weighted_log += rho[b] * np.log(np.maximum(s, 1e-6))

    log_c_raw = gaussian_filter(weighted_log, sigma=_sigma_voxels(voxel_spacing_xyz),
                                mode="nearest")
    # Calibrate so median in gland is 1 (i.e. log == 0). exp is monotonic ⇒
    # log Z = median(log_c_raw) inside the gland.
    pm = prostate_mask > 0
    log_z = float(np.median(log_c_raw[pm])) if pm.any() else 0.0
    log_c_calib = (log_c_raw - log_z).astype(np.float32)
    c_calib = np.exp(np.clip(log_c_calib, -60.0, 60.0)).astype(np.float32)

    return CDIsResult(
        cdis=c_calib, log_cdis=log_c_calib, adc=adc,
        z_calibration=float(np.exp(log_z)) if abs(log_z) < 80 else float("inf"),
        rho=dict(rho),
    )


# ── ρ optimisation (CDIs_t) ───────────────────────────────────────────────────

def optimize_rho(per_patient_inputs: list[dict],
                 task: str = "csPCa_vs_healthy",
                 *, rho_init: Optional[dict[int, float]] = None,
                 b_synth: Iterable[int] = config.B_SYNTH,
                 b_ref: int = config.B_REF,
                 maxiter: int = 50,
                 verbose: bool = False) -> tuple[dict[int, float], float]:
    from scipy.optimize import minimize
    from sklearn.metrics import roc_auc_score

    bs = [b_ref] + list(b_synth)
    if rho_init is None:
        rho_init = {b: 1.0 for b in bs}
    x0 = np.array([rho_init[b] for b in bs], dtype=float)

    def _score(x: np.ndarray) -> float:
        rho = {b: float(v) for b, v in zip(bs, x)}
        labels: list[int] = []
        scores: list[float] = []
        for inp in per_patient_inputs:
            res = compute_cdis(inp["signals"], inp["prostate_mask"],
                               inp["voxel_spacing"], rho=rho,
                               b_synth=b_synth, b_ref=b_ref)
            log_c = res.log_cdis
            pm = inp["prostate_mask"] > 0
            les = inp["lesion_rois"]
            all_les = np.zeros_like(pm)
            for lm, _ in les:
                all_les |= (lm > 0)
            healthy_idx = np.argwhere(pm & ~all_les)
            rng = np.random.default_rng(config.RNG_SEED)
            if len(healthy_idx) > config.HEALTHY_VOXELS_PER_PATIENT:
                healthy_idx = healthy_idx[rng.choice(
                    len(healthy_idx), config.HEALTHY_VOXELS_PER_PATIENT, replace=False)]
            healthy_vals = log_c[healthy_idx[:, 0], healthy_idx[:, 1], healthy_idx[:, 2]]

            if task == "PCa_vs_healthy":
                labels += [0] * len(healthy_vals); scores += list(healthy_vals)
                for lm, _ in les:
                    m = lm > 0
                    if m.any():
                        labels.append(1); scores.append(float(np.median(log_c[m])))
            elif task == "csPCa_vs_healthy":
                labels += [0] * len(healthy_vals); scores += list(healthy_vals)
                for lm, is_cs in les:
                    if not is_cs:
                        continue
                    m = lm > 0
                    if m.any():
                        labels.append(1); scores.append(float(np.median(log_c[m])))
            elif task == "csPCa_vs_insPCa":
                for lm, is_cs in les:
                    m = lm > 0
                    if not m.any():
                        continue
                    labels.append(1 if is_cs else 0)
                    scores.append(float(np.median(log_c[m])))
            else:
                raise ValueError(task)

        if len(set(labels)) < 2:
            return 0.0
        return -float(roc_auc_score(labels, scores))

    res = minimize(_score, x0, method="Nelder-Mead",
                   options={"maxiter": maxiter, "xatol": 1e-3, "fatol": 1e-4,
                            "disp": verbose})
    return {int(b): float(v) for b, v in zip(bs, res.x)}, -float(res.fun)
