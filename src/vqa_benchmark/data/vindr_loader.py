from pathlib import Path

import numpy as np
import pydicom
from PIL import Image


def dicom_to_uint8_array(dicom_path: str) -> np.ndarray:
    ds = pydicom.dcmread(dicom_path)
    arr = ds.pixel_array.astype(np.float32)

    if getattr(ds, "PhotometricInterpretation", "") == "MONOCHROME1":
        arr = arr.max() - arr

    arr = arr - arr.min()

    if arr.max() > 0:
        arr = arr / arr.max()

    arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
    return arr


def save_dicom_as_png(dicom_path: str, output_path: str) -> str:
    arr = dicom_to_uint8_array(dicom_path)
    image = Image.fromarray(arr).convert("RGB")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)

    return str(output_path)
