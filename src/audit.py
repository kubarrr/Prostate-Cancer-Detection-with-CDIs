"""Audit data completeness across the PROSTATEx cohort."""
from __future__ import annotations

import csv
import sys
from pathlib import Path

from . import config, data_io


def audit_patient(pid: str, classes) -> dict:
    pdir = Path(config.DWI_DICOM_DIR) / pid
    series = data_io.list_series(pdir)
    dwi = data_io.find_dwi_series(series)

    bvals: set[int] = set()
    if dwi is not None:
        bvals = data_io.probe_dwi_bvalues(dwi)

    n_les_t2  = sum(1 for _ in config.LESION_T2_DIR.glob(f"{pid}-Finding*.nii.gz"))
    n_les_adc = sum(1 for _ in config.LESION_ADC_DIR.glob(f"{pid}-Finding*.nii.gz"))
    n_in_csv  = int((classes["pid"] == pid).sum())

    return {
        "pid": pid,
        "has_DWI": dwi is not None,
        "DWI_desc": dwi.description if dwi else "",
        "DWI_bvals": ",".join(str(b) for b in sorted(bvals)),
        "DWI_has_50_400_800": {50, 400, 800}.issubset(bvals),
        "has_T2_nifti":  any(config.T2_IMAGE_DIR.glob(f"{pid}_t2_tse_tra_*.nii.gz")),
        "has_ADC_nifti": any(config.ADC_IMAGE_DIR.glob(f"{pid}_ep2d_diff_tra_*.nii.gz")),
        "has_prostate_mask": (config.PROSTATE_MASK_DIR / f"{pid}.nii.gz").exists(),
        "has_pz_mask":       (config.PZ_MASK_DIR       / f"{pid}_pz.nii.gz").exists(),
        "has_tz_mask":       (config.TZ_MASK_DIR       / f"{pid}_tz.nii.gz").exists(),
        "n_lesion_masks_T2":  n_les_t2,
        "n_lesion_masks_ADC": n_les_adc,
        "n_findings_in_csv":  n_in_csv,
    }


def _is_complete(row: dict) -> bool:
    return bool(row["has_DWI"] and row["DWI_has_50_400_800"]
                and row["has_T2_nifti"] and row["has_ADC_nifti"]
                and row["has_prostate_mask"] and row["n_lesion_masks_T2"] > 0
                and row["n_findings_in_csv"] > 0)


def main() -> int:
    classes = data_io.load_classes()
    pids_dicom = {p.name for p in Path(config.DWI_DICOM_DIR).iterdir() if p.is_dir()}
    pids_masks = {p.stem.replace(".nii", "") for p in config.PROSTATE_MASK_DIR.glob("ProstateX-*.nii.gz")}
    pids = sorted(pids_dicom | pids_masks | set(classes["pid"].dropna().unique()))

    rows = []
    for i, pid in enumerate(pids, 1):
        rows.append(audit_patient(pid, classes))
        if i % 25 == 0:
            print(f"  audited {i}/{len(pids)}", file=sys.stderr)

    for r in rows:
        r["complete"] = _is_complete(r)

    config.OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_csv = config.OUT_DIR / "patient_audit.csv"
    with out_csv.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    n_total = len(rows)
    n_complete = sum(1 for r in rows if r["complete"])
    print(f"Total patients seen: {n_total}")
    print(f"Complete:            {n_complete}")
    for k in ("has_DWI", "DWI_has_50_400_800", "has_T2_nifti", "has_ADC_nifti",
              "has_prostate_mask"):
        print(f"  {k}: {sum(1 for r in rows if r[k])}")
    print(f"  has_any_lesion_mask: {sum(1 for r in rows if r['n_lesion_masks_T2'] > 0)}")
    print(f"\nWrote {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
