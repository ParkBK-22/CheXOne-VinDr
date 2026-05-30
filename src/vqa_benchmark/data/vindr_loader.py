from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def dicom_to_uint8_array(dicom_path: str) -> np.ndarray:
    """
    VinDr-CXR common conversion rule:
    percentile 1-99 windowing -> 8-bit uint8.

    This follows the shared benchmark rule so all models use the same PNG input.
    """
    ds = pydicom.dcmread(dicom_path)
    img = ds.pixel_array.astype(np.float32)

    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        img = img.max() - img

    lo, hi = np.percentile(img, [1, 99])
    img = np.clip((img - lo) / (hi - lo + 1e-6), 0, 1)
    img = (img * 255.0).astype(np.uint8)

    return img


def save_dicom_as_png(
    dicom_path: str,
    output_path: str,
    long_side: int = 1024,
) -> str:
    """
    Save DICOM as RGB PNG using the shared benchmark rule:
    percentile 1-99 windowing -> long side 1024 -> RGB.
    """
    arr = dicom_to_uint8_array(dicom_path)
    im = Image.fromarray(arr).convert("L")

    w, h = im.size
    scale = long_side / max(w, h)
    new_w = round(w * scale)
    new_h = round(h * scale)

    im = im.resize((new_w, new_h)).convert("RGB")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(output_path)

    return str(output_path)


def resolve_vindr_dicom_path(image_root: str, image_id: str, split: str = "test") -> Path:
    """
    Resolve VinDr image path from image_id.

    Expected:
      image_root/test/{image_id}.dicom
    """
    return Path(image_root) / split / f"{image_id}.dicom"


def resolve_cached_png_path(
    png_root: str,
    image_id: str,
    split: str = "test",
) -> Path:
    return Path(png_root) / split / f"{image_id}.png"


def ensure_png_from_dicom(
    dicom_path: str,
    png_path: str,
    long_side: int = 1024,
) -> str:
    png_path = Path(png_path)

    if png_path.exists():
        return str(png_path)

    return save_dicom_as_png(
        dicom_path=dicom_path,
        output_path=str(png_path),
        long_side=long_side,
    )
