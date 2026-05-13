"""Data loading: DICOM DWI (per b-value) + NIfTI images/masks from PROSTATEx_masks."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import pydicom
import SimpleITK as sitk

from . import config


# ── DICOM series discovery ────────────────────────────────────────────────────

@dataclass
class SeriesInfo:
    description: str
    series_dir: Path
    files: list[Path]


def list_series(patient_dir: Path) -> list[SeriesInfo]:
    out: list[SeriesInfo] = []
    if not patient_dir.is_dir():
        return out
    for study in patient_dir.iterdir():
        if not study.is_dir():
            continue
        for series in study.iterdir():
            if not series.is_dir():
                continue
            files = sorted(series.glob("*.dcm"))
            if not files:
                continue
            try:
                ds = pydicom.dcmread(str(files[0]), stop_before_pixels=True)
            except Exception:
                continue
            desc = str(getattr(ds, "SeriesDescription", "") or "")
            out.append(SeriesInfo(description=desc, series_dir=series, files=files))
    return out


def find_dwi_series(series_list: list[SeriesInfo]) -> Optional[SeriesInfo]:
    for s in series_list:
        d = s.description.lower()
        if any(p in d for p in ("adc", "calc_bval")):
            continue
        if any(p.lower() in d for p in config.DWI_DESCS):
            return s
    return None


# ── DICOM b-value handling ────────────────────────────────────────────────────

_SIEMENS_BVAL = (0x0019, 0x100c)
_STD_BVAL     = (0x0018, 0x9087)


def read_bvalue(ds) -> Optional[int]:
    for tag in (_SIEMENS_BVAL, _STD_BVAL):
        try:
            v = ds[tag].value
            if isinstance(v, bytes):
                v = v.decode("ascii", errors="ignore")
            if isinstance(v, str):
                v = float(v.strip())
            return int(v)
        except Exception:
            continue
    return None


def _slice_position(ds) -> float:
    iop = np.asarray(ds.ImageOrientationPatient, dtype=float)
    ipp = np.asarray(ds.ImagePositionPatient,   dtype=float)
    normal = np.cross(iop[:3], iop[3:])
    return float(np.dot(normal, ipp))


def load_dwi_per_bvalue(series: SeriesInfo, b_values: Iterable[int]
                        ) -> tuple[dict[int, np.ndarray], sitk.Image]:
    """Group multi-b DWI files into one 3D volume per b-value.

    Returns (signals[b] -> (Z,Y,X) float32, ref_img on the b=min grid).
    """
    b_target = set(int(b) for b in b_values)
    by_b: dict[int, list[tuple[float, str]]] = {}

    for f in series.files:
        ds = pydicom.dcmread(str(f), stop_before_pixels=True)
        b = read_bvalue(ds)
        if b is None or b not in b_target:
            continue
        by_b.setdefault(b, []).append((_slice_position(ds), str(f)))

    if not by_b:
        raise RuntimeError(f"No b-values from {sorted(b_target)} in {series.series_dir}")

    signals: dict[int, np.ndarray] = {}
    ref_img: Optional[sitk.Image] = None

    for b, items in by_b.items():
        items.sort(key=lambda t: t[0])
        reader = sitk.ImageSeriesReader()
        reader.SetFileNames([f for _, f in items])
        img = reader.Execute()
        if img.GetNumberOfComponentsPerPixel() > 1:
            img = sitk.VectorIndexSelectionCast(img, 0)
        # Floor so log() is safe downstream.
        signals[b] = np.maximum(sitk.GetArrayFromImage(img).astype(np.float32), 1.0)
        if ref_img is None or b == min(by_b):
            ref_img = img

    return signals, ref_img


def probe_dwi_bvalues(series: SeriesInfo) -> set[int]:
    out: set[int] = set()
    for f in series.files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True)
            b = read_bvalue(ds)
            if b is not None:
                out.add(int(b))
        except Exception:
            continue
    return out


# ── NIfTI loaders (Cuocolo PROSTATEx_masks) ───────────────────────────────────

def _first_glob(directory: Path, pattern: str) -> Optional[Path]:
    hits = sorted(directory.glob(pattern))
    return hits[0] if hits else None


def load_t2_image(pid: str) -> Optional[sitk.Image]:
    p = _first_glob(config.T2_IMAGE_DIR, f"{pid}_t2_tse_tra_*.nii.gz")
    return sitk.ReadImage(str(p)) if p else None


def load_adc_image(pid: str) -> Optional[sitk.Image]:
    p = _first_glob(config.ADC_IMAGE_DIR, f"{pid}_ep2d_diff_tra_*.nii.gz")
    return sitk.ReadImage(str(p)) if p else None


def load_prostate_mask(pid: str) -> Optional[sitk.Image]:
    p = config.PROSTATE_MASK_DIR / f"{pid}.nii.gz"
    return sitk.ReadImage(str(p)) if p.exists() else None


def load_zone_mask(pid: str, zone: str) -> Optional[sitk.Image]:
    z = zone.lower()
    base = {"pz": config.PZ_MASK_DIR, "tz": config.TZ_MASK_DIR}[z]
    p = base / f"{pid}_{z}.nii.gz"
    return sitk.ReadImage(str(p)) if p.exists() else None


@dataclass
class LesionMask:
    pid: str
    finding_id: int          # 0, 1, 2, ...
    t2_path: Optional[Path]
    adc_path: Optional[Path]
    is_csPCa: bool
    ggg: Optional[int]       # ISUP/Gleason Grade Group, None if no biopsy


_LESION_NAME_RE = re.compile(r"(?P<pid>ProstateX-\d+)-Finding(?P<fid>\d+)-")


def _index_lesion_masks(directory: Path) -> dict[tuple[str, int], Path]:
    """Pick one mask per (pid, finding) — prefer axial t2_tse_tra over coronal/variants."""
    candidates: dict[tuple[str, int], list[Path]] = {}
    for p in directory.glob("ProstateX-*.nii.gz"):
        m = _LESION_NAME_RE.match(p.name)
        if not m:
            continue
        key = (m["pid"], int(m["fid"]))
        candidates.setdefault(key, []).append(p)

    def score(path: Path) -> tuple:
        n = path.name.lower()
        # lower tuple = preferred
        return (
            "_cor" in n,                    # coronal: deprioritise
            "grappa" in n,                  # non-default variant
            n.count("_roi.nii") > 1,        # double-suffixed typo
            n,                              # tie-break alphabetical
        )

    return {k: min(v, key=score) for k, v in candidates.items()}


def list_lesion_masks(pid: str, classes: pd.DataFrame) -> list[LesionMask]:
    """All lesions for one patient with class labels merged in."""
    t2_idx  = _index_lesion_masks(config.LESION_T2_DIR)
    adc_idx = _index_lesion_masks(config.LESION_ADC_DIR)

    fids = {k[1] for k in t2_idx if k[0] == pid} | {k[1] for k in adc_idx if k[0] == pid}
    out: list[LesionMask] = []
    for fid in sorted(fids):
        row = classes[(classes["pid"] == pid) & (classes["finding_id"] == fid)]
        if row.empty:
            continue
        out.append(LesionMask(
            pid=pid, finding_id=fid,
            t2_path  = t2_idx.get((pid, fid)),
            adc_path = adc_idx.get((pid, fid)),
            is_csPCa = bool(row["is_csPCa"].iloc[0]),
            ggg      = row["ggg"].iloc[0] if not pd.isna(row["ggg"].iloc[0]) else None,
        ))
    return out


# ── classes CSV ───────────────────────────────────────────────────────────────

def load_classes(csv_path: Optional[Path] = None) -> pd.DataFrame:
    """Parse PROSTATEx_Classes.csv into pid / finding_id / is_csPCa / ggg."""
    df = pd.read_csv(csv_path or config.CLASSES_CSV)
    df = df.rename(columns={
        "ID": "raw_id",
        "Clinically Significant": "is_csPCa",
        "Gleason Grade Group": "ggg",
    })
    parsed = df["raw_id"].str.extract(r"(?P<pid>ProstateX-\d+)_Finding(?P<finding_id>\d+)")
    df["pid"] = parsed["pid"]
    df["finding_id"] = parsed["finding_id"].astype(int)
    df["is_csPCa"] = df["is_csPCa"].astype(str).str.upper().eq("TRUE")
    df["ggg"] = pd.to_numeric(df["ggg"], errors="coerce")
    return df[["pid", "finding_id", "is_csPCa", "ggg", "raw_id"]]


def patient_ids_from_classes(df: pd.DataFrame) -> list[str]:
    return sorted(df["pid"].dropna().unique().tolist())
