#!/usr/bin/env python3
# ============================================================================
#  build_perturbations.py
#  VinDr-CXR visual-faithfulness perturbation set generator  (조건 A / B)
#  (팀 공유용 — DICOM 경로만 본인 환경에 맞게 --dicom-dir 로 바꾸면 됨)
# ----------------------------------------------------------------------------
#  "모델이 실제로 이미지를 보고 답하는가"를 검증하는 반사실(counterfactual) 이미지 생성.
#  병변을 정상 조직으로 치환한 뒤 같은 질문을 다시 물어, 답 변화율(flip rate)로
#  visual grounding vs 텍스트 prior 를 구분한다.
#
#  조건 (2개):
#    A (lesion-removal, positive)   : 단일 병변(bbox 1개) 이미지 272장의 병변 bbox 를
#                                     다른 정상 이미지(cross-donor)의 동일영역으로 치환.
#                                     → Q1 'Yes→No' flip 이 클수록 visual grounding.
#    B (normal-manip, neg. control) : 정상(No finding) 이미지의, A 의 bbox 분포에서 뽑은
#                                     영역을 다른 정상 donor 패치로 치환.
#                                     → Q1 'No' 유지가 정상. flip(No→Yes) = "조작 흔적
#                                       자체가 abnormal 로 오인되는 비율" = A 의 baseline.
#
#  핵심 규약:
#    [B1] 치환은 percentile 1–99 windowing → 8-bit 공간에서 수행(서로 다른 DICOM 의 raw
#         픽셀 범위 차이로 생기는 '밝기 박스' artifact 방지) → 장변 1024 → RGB.
#         (run_qwen/run_internvl 의 dicom_to_pil 과 동일 windowing. ※ apply_voi_lut 미적용
#          — 우리 Set1 baseline 과 일치. 효근 crop 파이프라인은 voi_lut 적용하므로 별개.)
#    [B2] donor 패치 = target bbox 의 정규화 좌표(0~1) 동일영역 → bbox 크기로 리사이즈 paste.
#    [B3] 조건 B 영역 = A 272개 bbox 의 정규화 분포에서 결정론적 샘플 (size/location-matched).
#    [B4] donor = 정상 풀에서 (seed, image_id) 결정론 random 1:1 (target≠donor).
#  완전 결정론 → 어느 머신/경로에서 돌려도 동일 perturbed PNG (cross-model 비교 전제).
#
#  질문 (Q1/Q3/Q5 만):  병변 제거 후에도 '정답 선지가 존재'하는 유형만 flip 지표로 사용.
#    Q1(Yes→No) · Q3(1→0) · Q5(→No specific disease).  Q2/Q4 는 'no abnormality' 선지가
#    없어 제거 후 ill-posed → 제외. (조건 B 는 정상이라 자연히 Q1(+Q3) 만 존재.)
#
#  산출물:
#    <out-dir>/A/<image_id>.png            조건 A perturbed (모델 입력용)
#    <out-dir>/B/<image_id>.png            조건 B perturbed
#    <out-dir>/original/<image_id>.png     windowed 원본 (--save-original; 매칭 baseline/plot)
#    <out-dir>/questions_A.jsonl           A 대상 이미지의 Q1/Q3/Q5 (qa_set1 레코드 그대로 필터)
#    <out-dir>/questions_B.jsonl           B 대상 이미지의 Q1/Q3 (정상이라 Q5 없음)
#    <out-dir>/perturbation_manifest.jsonl 재현·감사용
#
#  의존성: numpy, pydicom, pillow
#
#  실행 예:
#    python build_perturbations.py --dicom-dir <DICOM> --ann-csv <ANN.csv> \
#        --qa-jsonl <qa_set1_options.jsonl> --out-dir ./perturb_set --save-original
#    # sanity (각 10장): 위에 --n-a 10 --n-b 10 추가
# ============================================================================
import argparse, csv, json, os, random, collections
import numpy as np, pydicom
from PIL import Image

NO_FINDING = "No finding"


def dicom_to_windowed_8bit(path):
    """DICOM → percentile 1–99 windowing → 8-bit 2D uint8 (원본 해상도). MONOCHROME1 반전 포함.
    run_qwen/run_internvl 의 dicom_to_pil 과 동일 windowing (apply_voi_lut 미적용)."""
    ds = pydicom.dcmread(path)
    a = ds.pixel_array.astype(np.float32)
    if ds.get('PhotometricInterpretation', 'MONOCHROME2') == 'MONOCHROME1':
        a = a.max() - a
    lo, hi = np.percentile(a, [1, 99])
    a = np.clip((a - lo) / (hi - lo + 1e-6), 0, 1) * 255
    return a.astype(np.uint8)                       # H x W


def dicom_dims(path):
    """픽셀 디코드 없이 (Rows, Columns) 만 — A bbox 정규화 분포 산출용(빠름)."""
    ds = pydicom.dcmread(path, stop_before_pixels=True)
    return int(ds.Rows), int(ds.Columns)            # H, W


def to_rgb_long(arr8, long_side=1024):
    im = Image.fromarray(arr8).convert('RGB')
    w, h = im.size
    s = long_side / max(w, h)
    if s < 1:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))))
    return im


def load_annotations(ann_csv):
    """image_id -> [(class_name, [x_min,y_min,x_max,y_max] | None), ...]."""
    per = {}
    with open(ann_csv) as f:
        for row in csv.DictReader(f):
            img, cls = row["image_id"], row["class_name"]
            box = None
            if cls != NO_FINDING and row.get("x_min") not in (None, "", "NaN"):
                try:
                    box = [float(row["x_min"]), float(row["y_min"]),
                           float(row["x_max"]), float(row["y_max"])]
                except ValueError:
                    box = None
            per.setdefault(img, []).append((cls, box))
    return per


def paste_region(target8, donor8, box_t):
    """target8 의 box_t([x0,y0,x1,y1] 픽셀) 영역을 donor8 의 동일 정규화영역 패치로 치환
    (8-bit space). (out_array, used_box[clamped]) 반환."""
    Ht, Wt = target8.shape
    Hd, Wd = donor8.shape
    x0, y0, x1, y1 = box_t
    x0, y0 = max(0, int(round(x0))), max(0, int(round(y0)))
    x1, y1 = min(Wt, int(round(x1))), min(Ht, int(round(y1)))
    if x1 <= x0 or y1 <= y0:
        return target8.copy(), None
    bw, bh = x1 - x0, y1 - y0
    dx0, dy0 = int(round(x0 / Wt * Wd)), int(round(y0 / Ht * Hd))
    dx1, dy1 = int(round(x1 / Wt * Wd)), int(round(y1 / Ht * Hd))
    dx0, dy0 = max(0, dx0), max(0, dy0)
    dx1, dy1 = min(Wd, max(dx0 + 1, dx1)), min(Hd, max(dy0 + 1, dy1))
    patch = Image.fromarray(donor8[dy0:dy1, dx0:dx1]).resize((bw, bh))
    out = target8.copy()
    out[y0:y1, x0:x1] = np.asarray(patch)
    return out, [x0, y0, x1, y1]


def emit_questions(qa_jsonl, image_ids, qtypes, out_path):
    """qa_set1 에서 image_ids ∩ qtypes 질문만 골라 그대로(스키마 보존) 기록.
    효근 crop 파이프라인의 single_bbox_questions.jsonl 과 동일 형식 → 같은 추론 스크립트 호환."""
    ids = set(image_ids)
    qts = set(qtypes)
    dist = collections.Counter()
    n = 0
    with open(out_path, "w") as w:
        for l in open(qa_jsonl):
            q = json.loads(l)
            if q["image_id"] in ids and q["qtype"] in qts:
                w.write(json.dumps(q, ensure_ascii=False) + "\n")
                dist[q["qtype"]] += 1
                n += 1
    return n, dict(sorted(dist.items()))


def main():
    ap = argparse.ArgumentParser(
        description="VinDr-CXR perturbation generator (A: lesion-removal / B: normal-manip control)")
    ap.add_argument("--dicom-dir", default="/workspace/physionet.org/files/vindr-cxr/1.0.0/test",
                    help="DICOM 디렉토리 ({image_id}.<ext>). ★본인 경로로 바꾸면 됨.")
    ap.add_argument("--ann-csv", default="/workspace/physionet.org/files/vindr-cxr/1.0.0/annotations/annotations_test.csv")
    ap.add_argument("--qa-jsonl", default="/workspace/qa_set1_options.jsonl",
                    help="qa_set1_options.jsonl — questions_A/B.jsonl 필터용")
    ap.add_argument("--out-dir", default="/workspace/perturb_set")
    ap.add_argument("--condition", choices=["A", "B", "both"], default="both")
    ap.add_argument("--qtypes", nargs="+", default=["Q1_abnormality", "Q3_count", "Q5_disease"],
                    help="재추론할 qtype (기본 Q1/Q3/Q5 — 제거 후 정답 선지 존재하는 유형)")
    ap.add_argument("--n-a", type=int, default=272, help="조건 A 이미지 수 (sanity=10)")
    ap.add_argument("--n-b", type=int, default=272, help="조건 B 이미지 수 (sanity=10)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--long-side", type=int, default=1024)
    ap.add_argument("--ext", default="dicom", help="DICOM 파일 확장자")
    ap.add_argument("--save-original", action="store_true", help="windowed 원본 PNG도 저장")
    args = ap.parse_args()

    per = load_annotations(args.ann_csv)
    single, normal = [], []
    for img, items in per.items():
        boxes = [(c, b) for c, b in items if b is not None]
        if len(boxes) == 1:
            single.append((img, boxes[0][0], boxes[0][1]))
        elif len(boxes) == 0:
            normal.append(img)
    single.sort(key=lambda t: t[0])
    normal.sort()
    print(f"[scan] single-bbox(A pool)={len(single)}  normal(B/donor pool)={len(normal)}")

    a_targets = single[:args.n_a]
    b_targets = normal[:args.n_b]
    donor_pool = normal

    os.makedirs(args.out_dir, exist_ok=True)
    if args.condition in ("A", "both"):
        os.makedirs(f"{args.out_dir}/A", exist_ok=True)
    if args.condition in ("B", "both"):
        os.makedirs(f"{args.out_dir}/B", exist_ok=True)
    if args.save_original:
        os.makedirs(f"{args.out_dir}/original", exist_ok=True)

    # [B3] A 전체 정규화 bbox 분포 (픽셀 디코드 없이 dims 만으로).
    a_norm_full = []
    for img, finding, box in single:
        try:
            Ht, Wt = dicom_dims(f"{args.dicom_dir}/{img}.{args.ext}")
        except Exception as e:
            print(f"  [warn] dims skip {img}: {e}"); continue
        a_norm_full.append([box[0] / Wt, box[1] / Ht, box[2] / Wt, box[3] / Ht])
    print(f"[B3] A 정규화 bbox 분포 {len(a_norm_full)}개 (조건 B 영역 샘플 소스)")

    def pick_donor(target_id, tag):
        rng = random.Random(f"{args.seed}_{tag}_{target_id}")
        d = rng.choice(donor_pool)
        while d == target_id:
            d = rng.choice(donor_pool)
        return d

    mf = open(f"{args.out_dir}/perturbation_manifest.jsonl", "w")

    # ---------- 조건 A : cross-donor lesion-removal ----------
    if args.condition in ("A", "both"):
        n_ok = 0
        for img, finding, box in a_targets:
            try:
                t8 = dicom_to_windowed_8bit(f"{args.dicom_dir}/{img}.{args.ext}")
                donor = pick_donor(img, "A")
                d8 = dicom_to_windowed_8bit(f"{args.dicom_dir}/{donor}.{args.ext}")
            except Exception as e:
                print(f"  [warn] A skip {img}: {e}"); continue
            Ht, Wt = t8.shape
            out8, used = paste_region(t8, d8, box)
            to_rgb_long(out8, args.long_side).save(f"{args.out_dir}/A/{img}.png")
            if args.save_original:
                to_rgb_long(t8, args.long_side).save(f"{args.out_dir}/original/{img}.png")
            mf.write(json.dumps({"image_id": img, "condition": "A", "finding": finding,
                                 "donor_id": donor, "bbox_orig": box, "used_box": used,
                                 "orig_hw": [Ht, Wt], "seed": args.seed}) + "\n")
            n_ok += 1
        print(f"[A] lesion-removal {n_ok}/{len(a_targets)} done")

    # ---------- 조건 B : normal-manip negative control ----------
    if args.condition in ("B", "both"):
        n_ok = 0
        for img in b_targets:
            try:
                t8 = dicom_to_windowed_8bit(f"{args.dicom_dir}/{img}.{args.ext}")
            except Exception as e:
                print(f"  [warn] B skip {img}: {e}"); continue
            Ht, Wt = t8.shape
            rng = random.Random(f"{args.seed}_Bregion_{img}")
            nb = rng.choice(a_norm_full)
            box = [nb[0] * Wt, nb[1] * Ht, nb[2] * Wt, nb[3] * Ht]
            donor = pick_donor(img, "B")
            try:
                d8 = dicom_to_windowed_8bit(f"{args.dicom_dir}/{donor}.{args.ext}")
            except Exception as e:
                print(f"  [warn] B donor skip {img}<-{donor}: {e}"); continue
            out8, used = paste_region(t8, d8, box)
            to_rgb_long(out8, args.long_side).save(f"{args.out_dir}/B/{img}.png")
            if args.save_original:
                to_rgb_long(t8, args.long_side).save(f"{args.out_dir}/original/{img}.png")
            mf.write(json.dumps({"image_id": img, "condition": "B", "finding": None,
                                 "donor_id": donor, "region_norm": nb, "used_box": used,
                                 "orig_hw": [Ht, Wt], "seed": args.seed}) + "\n")
            n_ok += 1
        print(f"[B] normal-manip(control) {n_ok}/{len(b_targets)} done")

    mf.close()

    # ---------- 질문 필터 (Q1/Q3/Q5, qa_set1 스키마 보존) ----------
    if os.path.isfile(args.qa_jsonl):
        if args.condition in ("A", "both"):
            n, d = emit_questions(args.qa_jsonl, [t[0] for t in a_targets], args.qtypes,
                                  f"{args.out_dir}/questions_A.jsonl")
            print(f"[Q] questions_A.jsonl: {n}  {d}")
        if args.condition in ("B", "both"):
            n, d = emit_questions(args.qa_jsonl, b_targets, args.qtypes,
                                  f"{args.out_dir}/questions_B.jsonl")
            print(f"[Q] questions_B.jsonl: {n}  {d}")
    else:
        print(f"  [warn] qa-jsonl 없음({args.qa_jsonl}) → 질문 필터 생략")

    print(f"[done] out-dir = {args.out_dir}")


if __name__ == "__main__":
    main()
