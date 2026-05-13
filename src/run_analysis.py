"""End-to-end CDIs pipeline."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk

from . import analysis, cdis, config, data_io
from . import tissue_sampling as ts
from . import visualization as viz


# ── per-patient loader ────────────────────────────────────────────────────────

def load_patient(pid: str, classes: pd.DataFrame) -> Optional[ts.PatientBundle]:
    # DICOM DWI
    series = data_io.list_series(Path(config.DWI_DICOM_DIR) / pid)
    dwi_s = data_io.find_dwi_series(series)
    if dwi_s is None:
        return None
    signals, dwi_ref = data_io.load_dwi_per_bvalue(dwi_s, config.B_NATIVE)
    if not all(b in signals for b in config.B_NATIVE):
        return None

    # NIfTI inputs
    t2_img   = data_io.load_t2_image(pid)
    adc_img  = data_io.load_adc_image(pid)
    pm_img   = data_io.load_prostate_mask(pid)
    pz_img   = data_io.load_zone_mask(pid, "pz")
    tz_img   = data_io.load_zone_mask(pid, "tz")
    if pm_img is None or t2_img is None:
        return None

    # Resample everything onto the DWI grid.
    t2_on_dwi  = ts.to_array(ts.resample_to_reference(t2_img,  dwi_ref))
    adc_on_dwi = ts.to_array(ts.resample_to_reference(adc_img, dwi_ref)) if adc_img is not None else None
    pm_on_dwi  = ts.resample_mask(pm_img, dwi_ref).astype(np.uint8)
    pz_on_dwi  = ts.resample_mask(pz_img, dwi_ref) if pz_img is not None else None
    tz_on_dwi  = ts.resample_mask(tz_img, dwi_ref) if tz_img is not None else None

    # Lesion ROIs: prefer ADC-space mask (closer to DWI), fall back to T2-space.
    rois: list[tuple[np.ndarray, bool]] = []
    for les in data_io.list_lesion_masks(pid, classes):
        path = les.adc_path or les.t2_path
        if path is None:
            continue
        m = ts.resample_mask(sitk.ReadImage(str(path)), dwi_ref)
        m &= pm_on_dwi.astype(bool)
        if not m.any():
            continue
        rois.append((m, les.is_csPCa))

    return ts.PatientBundle(
        pid=pid, voxel_spacing=ts.voxel_spacing_xyz(dwi_ref),
        signals=signals, t2w=t2_on_dwi, adc=adc_on_dwi,
        prostate_mask=pm_on_dwi, pz_mask=pz_on_dwi, tz_mask=tz_on_dwi,
        lesion_rois=rois, ref=dwi_ref,
    )


def sample_patient(bundle: ts.PatientBundle, *,
                   rho: Optional[dict[int, float]] = None) -> dict:
    res = cdis.compute_cdis(bundle.signals, bundle.prostate_mask,
                            bundle.voxel_spacing, rho=rho)
    h_idx = ts.pick_healthy_indices(bundle.prostate_mask, bundle.lesion_rois)
    samples = {"CDIs": ts.sample_modality(res.log_cdis, h_idx, bundle.lesion_rois)}
    if bundle.t2w is not None:
        samples["T2w"] = ts.sample_modality(bundle.t2w, h_idx, bundle.lesion_rois)
    if bundle.adc is not None:
        samples["ADC"] = ts.sample_modality(bundle.adc, h_idx, bundle.lesion_rois)
    return {"pid": bundle.pid, "cdis_res": res, "healthy_idx": h_idx, "samples": samples}


# ── pipeline ──────────────────────────────────────────────────────────────────

def run(patients_subset: Optional[list[str]] = None,
        tune_rho_task: Optional[str] = None,
        save_overlay_for: Optional[str] = None,
        out_dir: Path = config.OUT_DIR) -> dict:
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    audit_csv = out_dir / "patient_audit.csv"
    if not audit_csv.exists():
        raise FileNotFoundError(f"{audit_csv} missing — run `python -m src.audit`")
    audit = pd.read_csv(audit_csv)
    complete_pids = audit.loc[audit["complete"], "pid"].tolist()
    if patients_subset is not None:
        complete_pids = [p for p in complete_pids if p in patients_subset]

    classes = data_io.load_classes()
    print(f"Complete patients to process: {len(complete_pids)}")

    bundles: list[ts.PatientBundle] = []
    t0 = time.time()
    for i, pid in enumerate(complete_pids, 1):
        try:
            b = load_patient(pid, classes)
            if b is None:
                print(f"  [{i:3d}] {pid}: skipped"); continue
            bundles.append(b)
            cs  = sum(1 for _, c in b.lesion_rois if c)
            ins = sum(1 for _, c in b.lesion_rois if not c)
            print(f"  [{i:3d}] {pid}: lesions={len(b.lesion_rois)} (cs={cs}, ins={ins})")
        except Exception as e:
            print(f"  [{i:3d}] {pid}: ERROR {type(e).__name__}: {e}")
    print(f"Loaded {len(bundles)} patients in {time.time()-t0:.1f}s")

    # CDIs_t (optional ρ tuning)
    rho_t = None
    if tune_rho_task is not None:
        rho_inputs = [{"signals": b.signals, "prostate_mask": b.prostate_mask,
                       "voxel_spacing": b.voxel_spacing, "lesion_rois": b.lesion_rois}
                      for b in bundles]
        print(f"\nTuning ρ for '{tune_rho_task}' on {len(rho_inputs)} patients...")
        rho_t, auc_t = cdis.optimize_rho(rho_inputs, task=tune_rho_task, maxiter=30)
        print(f"  best AUC={auc_t:.4f}  ρ={{ {', '.join(f'{b}:{round(v,3)}' for b,v in rho_t.items())} }}")

    # Sampling
    all_samples = []
    all_samples_t = []
    for b in bundles:
        s = sample_patient(b)
        all_samples.append(s["samples"])
        if rho_t is not None:
            res_t = cdis.compute_cdis(b.signals, b.prostate_mask, b.voxel_spacing, rho=rho_t)
            all_samples_t.append({"CDIs_t":
                ts.sample_modality(res_t.log_cdis, s["healthy_idx"], b.lesion_rois)})

    pools = [
        ts.pool_modality("CDIs", [s["CDIs"] for s in all_samples]),
        ts.pool_modality("T2w",  [s["T2w"]  for s in all_samples if "T2w" in s]),
        ts.pool_modality("ADC",  [s["ADC"]  for s in all_samples if "ADC" in s]),
    ]
    if rho_t is not None:
        pools.append(ts.pool_modality("CDIs_t", [s["CDIs_t"] for s in all_samples_t]))

    print("\nWriting figures...")
    viz.plot_histograms(pools, save_path=out_dir / "fig1_histograms.png",
                        title="Figure 1 — tissue distributions")
    viz.plot_roc_grid(pools, save_path=out_dir / "fig2_roc.png",
                      title="Figure 2 — ROC by task")
    table = viz.auc_table(pools)
    table.to_csv(out_dir / "auc_table.csv", index=False)
    print("\nAUC summary:"); print(table.to_string(index=False))

    overlay_pid = save_overlay_for
    if overlay_pid is None and bundles:
        for b in bundles:
            if any(is_cs for _, is_cs in b.lesion_rois):
                overlay_pid = b.pid; break
    if overlay_pid is not None:
        b = next((x for x in bundles if x.pid == overlay_pid), None)
        if b is not None:
            res = cdis.compute_cdis(b.signals, b.prostate_mask, b.voxel_spacing)
            viz.plot_patient_overlays(
                b.t2w, b.adc, res.cdis, b.prostate_mask, b.lesion_rois,
                save_path=out_dir / f"fig3_overlay_{b.pid}.png",
                title=f"Figure 3 example — {b.pid}")
            print(f"Saved overlay for {b.pid}")

    return {"pools": pools, "rho_t": rho_t, "table": table, "n_patients": len(bundles)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patients", type=int, default=None)
    ap.add_argument("--tune-rho", type=str, default=None,
                    choices=["csPCa_vs_healthy", "PCa_vs_healthy", "csPCa_vs_insPCa"])
    ap.add_argument("--overlay", type=str, default=None)
    args = ap.parse_args(argv)

    subset = None
    if args.patients is not None:
        audit = pd.read_csv(config.OUT_DIR / "patient_audit.csv")
        subset = audit.loc[audit["complete"], "pid"].head(args.patients).tolist()

    run(patients_subset=subset, tune_rho_task=args.tune_rho,
        save_overlay_for=args.overlay)
    return 0


if __name__ == "__main__":
    sys.exit(main())
