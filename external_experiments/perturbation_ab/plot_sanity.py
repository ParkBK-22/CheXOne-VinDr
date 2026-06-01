#!/usr/bin/env python3
# ============================================================================
#  plot_sanity.py — perturbation sanity 시각화
#  build_perturbations.py 산출물(+ qa_set1)을 받아, 조건별로
#    [원본(치환영역 빨강박스)] | [조작본] + 해당 이미지의 Set1 질문/GT/기대답
#  을 한 패널로 그려 눈으로 검수한다.  조건별 figure 1장씩 저장.
#
#  실행:
#    python plot_sanity.py --out-dir ./perturb_sanity \
#        --qa /workspace/qa_set1_options.jsonl --save-dir ./perturb_sanity
# ============================================================================
import argparse, json, collections
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

# 조건 A(병변 제거) 후 기대답(counterfactual). B 는 control 이라 원본 GT 유지가 기대.
# (plot 텍스트는 서버 matplotlib 기본 폰트 한글 미지원 → ASCII 로 표기)
EXPECT_A = {
    "Q1_abnormality": "No",
    "Q3_count": "0",
    "Q5_disease": "No specific disease",
    "Q2_finding": "(expect change)",
    "Q4_location": "(expect change)",
}


def load_qa_by_image(qa_path):
    by = collections.defaultdict(list)
    for l in open(qa_path):
        r = json.loads(l)
        by[r["image_id"]].append(r)
    return by


def draw_box(ax, used_box, orig_hw, png_size):
    """used_box(원본 픽셀좌표)를 1024-리사이즈된 PNG 좌표로 스케일해 빨강 박스."""
    if not used_box:
        return
    Ht, Wt = orig_hw
    pw, ph = png_size
    sx, sy = pw / Wt, ph / Ht
    x0, y0, x1, y1 = used_box
    ax.add_patch(Rectangle((x0 * sx, y0 * sy), (x1 - x0) * sx, (y1 - y0) * sy,
                           fill=False, edgecolor="red", linewidth=1.5))


def fig_for(cond, rows, qa_by, out_dir, save_path):
    n = len(rows)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3.4 * n), squeeze=False)
    for i, m in enumerate(rows):
        img = m["image_id"]
        orig = Image.open(f"{out_dir}/original/{img}.png")
        pert = Image.open(f"{out_dir}/{cond}/{img}.png")
        ax0, ax1 = axes[i]
        ax0.imshow(orig); ax0.axis("off")
        ax0.set_title(f"original  {img[:12]}...", fontsize=8)
        draw_box(ax0, m.get("used_box"), m["orig_hw"], orig.size)
        ax1.imshow(pert); ax1.axis("off")
        draw_box(ax1, m.get("used_box"), m["orig_hw"], pert.size)

        head = (f"[{cond}] " + (f"finding={m.get('finding')}  " if cond.startswith("A") else "normal  ")
                + f"donor={str(m.get('donor_id'))[:12]}...")
        lines = [head]
        for q in sorted(qa_by.get(img, []), key=lambda r: r["qid"]):
            qt, ans = q["qtype"], q["answer"]
            exp = EXPECT_A.get(qt, ans) if cond.startswith("A") else f"{ans} (control: keep)"
            lines.append(f"{qt}: GT={ans}  -> expect {exp}")
        if len(qa_by.get(img, [])) == 0:
            lines.append("(no Set1 questions for this image)")
        ax1.set_title("\n".join(lines), fontsize=7, loc="left")
    cond_name = {"A": "A: lesion-removal (cross-donor)",
                 "B": "B: normal-manip (negative control)"}[cond]
    fig.suptitle(f"Perturbation sanity - condition {cond_name}", fontsize=11, y=1.0)
    plt.tight_layout()
    plt.savefig(save_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print("saved", save_path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="/workspace/perturb_sanity")
    ap.add_argument("--qa", default="/workspace/qa_set1_options.jsonl")
    ap.add_argument("--save-dir", default=None)
    args = ap.parse_args()
    save_dir = args.save_dir or args.out_dir

    manifest = [json.loads(l) for l in open(f"{args.out_dir}/perturbation_manifest.jsonl")]
    qa_by = load_qa_by_image(args.qa)
    for cond in ("A", "B"):
        rows = [m for m in manifest if m["condition"] == cond and not m.get("skipped")]
        if rows:
            fig_for(cond, rows, qa_by, args.out_dir, f"{save_dir}/sanity_{cond}.png")


if __name__ == "__main__":
    main()
