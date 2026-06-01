"""Step 1 — VinDr test set에서 (a) single-bbox 이미지 추출 (b) 관련 질문 필터 (c) original PNG + crop PNG 생성.

Usage:
    python 01_prepare_dataset.py \
        --vindr_root /workspace/physionet.org/files/vindr-cxr/1.0.0 \
        --qa_jsonl   /workspace/qa_set1_options.jsonl \
        --out_dir    /workspace/pipeline_out \
        --margins    1.0 1.5 2.0

생성물:
  out_dir/
  ├── single_bbox_info.csv          # image_id, class_name, x_min, y_min, x_max, y_max
  ├── single_bbox_questions.jsonl   # qa_set1 중 single-bbox image의 질문만
  ├── png_original/{image_id}.png   # kit spec (장변 1024, percentile 1-99, RGB)
  └── png_crop/margin_{m}/{image_id}.png  # bbox 중심 m배 crop
"""
import os, argparse, json, time
import pandas as pd
import numpy as np
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from PIL import Image


def dicom_window(path):
    """DICOM → uint8 grayscale (kit spec: percentile 1–99 windowing)."""
    ds = pydicom.dcmread(path)
    arr = ds.pixel_array.astype(np.float32)
    try:
        arr = apply_voi_lut(arr, ds)
    except Exception:
        pass
    if ds.get("PhotometricInterpretation", "MONOCHROME2") == "MONOCHROME1":
        arr = arr.max() - arr
    lo, hi = np.percentile(arr, [1, 99])
    arr = np.clip((arr - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return arr.astype(np.uint8)


def save_original_png(img, out_path, long_side=1024):
    """장변 long_side로 비율 유지 resize → 3채널 RGB PNG (kit spec)."""
    im = Image.fromarray(img).convert("L")
    w, h = im.size
    s = long_side / max(w, h)
    im.resize((round(w * s), round(h * s))).convert("RGB").save(out_path)


def crop_with_margin(img, x_min, y_min, x_max, y_max, margin):
    """bbox 중심을 유지하면서 (w*margin, h*margin) 영역으로 crop (이미지 경계 clip)."""
    H, W = img.shape[:2]
    cx = (x_min + x_max) / 2.0
    cy = (y_min + y_max) / 2.0
    new_w = (x_max - x_min) * margin
    new_h = (y_max - y_min) * margin
    x0 = int(max(0, round(cx - new_w / 2)))
    y0 = int(max(0, round(cy - new_h / 2)))
    x1 = int(min(W, round(cx + new_w / 2)))
    y1 = int(min(H, round(cy + new_h / 2)))
    return img[y0:y1, x0:x1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vindr_root", required=True,
                    help="VinDr 1.0.0 root: 안에 test/, annotations/ 있어야 함")
    ap.add_argument("--qa_jsonl", required=True,
                    help="qa_set1_options.jsonl (5,293 MC 질문)")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--qtypes", type=str, nargs="+",
                    default=["Q1_abnormality", "Q2_finding", "Q5_disease"],
                    help="추출할 qtype 들 (기본: Q1/Q2/Q5 — crop 비교에 의미 있는 것만. "
                         "Q3 trivial=1, Q4 crop에선 frame 손실).")
    ap.add_argument("--long_side", type=int, default=1024,
                    help="원본 PNG 장변 (kit spec=1024)")
    args = ap.parse_args()

    test_dicom_dir = os.path.join(args.vindr_root, "test")
    ann_csv = os.path.join(args.vindr_root, "annotations", "annotations_test.csv")
    assert os.path.isdir(test_dicom_dir), f"missing: {test_dicom_dir}"
    assert os.path.isfile(ann_csv), f"missing: {ann_csv}"

    os.makedirs(args.out_dir, exist_ok=True)
    png_orig_dir = os.path.join(args.out_dir, "png_original")
    os.makedirs(png_orig_dir, exist_ok=True)
    for m in args.margins:
        os.makedirs(os.path.join(args.out_dir, "png_crop", f"margin_{m}"), exist_ok=True)

    # --- (1) 단일 bbox 이미지 추출 ---
    ann = pd.read_csv(ann_csv)
    abn = ann[ann["class_name"] != "No finding"].dropna(subset=["x_min"]).copy()
    counts = abn.groupby("image_id").size()
    single_ids = counts[counts == 1].index.tolist()
    single_df = abn[abn["image_id"].isin(single_ids)].copy()
    print(f"[1] single-bbox images: {len(single_df)}")
    single_df.to_csv(os.path.join(args.out_dir, "single_bbox_info.csv"), index=False)

    # --- (2) qa_set1에서 단일 bbox 이미지 + 선택한 qtype 질문만 필터 ---
    single_id_set = set(single_ids)
    qtype_set = set(args.qtypes)
    qa = [json.loads(l) for l in open(args.qa_jsonl)]
    qa_sub = [q for q in qa if q["image_id"] in single_id_set and q["qtype"] in qtype_set]
    with open(os.path.join(args.out_dir, "single_bbox_questions.jsonl"), "w") as f:
        for q in qa_sub: f.write(json.dumps(q) + "\n")
    from collections import Counter
    qtype_dist = Counter(q["qtype"] for q in qa_sub)
    print(f"[2] filtered questions: {len(qa_sub)}  (qtypes={args.qtypes})")
    for qt in ["Q1_abnormality","Q2_finding","Q3_count","Q4_location","Q5_disease"]:
        if qt in qtype_set:
            print(f"    {qt}: {qtype_dist.get(qt, 0)}")

    # --- (3) PNG 생성: original (1024) + crops ---
    t0 = time.time()
    n_ok = n_fail = 0
    for i, row in single_df.iterrows():
        iid = row["image_id"]
        dpath = os.path.join(test_dicom_dir, f"{iid}.dicom")
        if not os.path.exists(dpath):
            n_fail += 1; continue
        try:
            img_full = dicom_window(dpath)  # 풀 해상도 grayscale
        except Exception as e:
            print(f"  [fail] {iid}: {e}", flush=True); n_fail += 1; continue
        # original (long_side resize)
        out_orig = os.path.join(png_orig_dir, f"{iid}.png")
        save_original_png(img_full, out_orig, long_side=args.long_side)
        # crops (각 margin)
        for m in args.margins:
            crop = crop_with_margin(img_full, row.x_min, row.y_min, row.x_max, row.y_max, m)
            if crop.size == 0: continue
            out = os.path.join(args.out_dir, "png_crop", f"margin_{m}", f"{iid}.png")
            Image.fromarray(crop).convert("RGB").save(out)
        n_ok += 1
        if n_ok % 50 == 0:
            el = time.time() - t0
            print(f"  [{n_ok}/{len(single_df)}] elapsed={el:.0f}s rate={n_ok/el:.2f}/s", flush=True)

    print(f"[3] PNG done: ok={n_ok}  fail={n_fail}  in {time.time()-t0:.0f}s")
    print(f"\noutputs:")
    print(f"  {args.out_dir}/single_bbox_info.csv")
    print(f"  {args.out_dir}/single_bbox_questions.jsonl  ({len(qa_sub)} 질문)")
    print(f"  {args.out_dir}/png_original/  ({n_ok} PNG)")
    for m in args.margins:
        print(f"  {args.out_dir}/png_crop/margin_{m}/")


if __name__ == "__main__":
    main()
