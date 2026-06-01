import argparse
import json
import re
from pathlib import Path

from tqdm import tqdm

from vqa_benchmark.models import build_model


def normalize(text):
    if text is None:
        return ""
    text = str(text).strip().lower()
    text = re.sub(r"[^a-z0-9\s/_.-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def map_option_letter(pred, options):
    if not options:
        return str(pred)

    raw = str(pred).strip()
    pred_norm = normalize(raw)

    for opt in options:
        if normalize(opt) == pred_norm:
            return str(opt)

    m = re.search(r"\b([A-Ka-k])\b", raw)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    m = re.match(r"^\s*[\(\[]?([A-Ka-k])[\)\]\.:]?", raw)
    if m:
        idx = ord(m.group(1).upper()) - ord("A")
        if 0 <= idx < len(options):
            return str(options[idx])

    return raw


def build_prompt(q):
    question = q["question"]
    options = q.get("options")

    if options:
        option_lines = "\n".join(
            f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(options)
        )
        return (
            f"{question}\n\n"
            f"Options:\n"
            f"{option_lines}\n\n"
            f"Answer with only the option text.\n\n"
            f"Answer briefly."
        )

    return f"{question}\n\nAnswer briefly."


def is_correct(q, pred):
    gt = normalize(q.get("answer", ""))
    qtype = q.get("qtype", "")

    pred_text = map_option_letter(pred, q.get("options"))
    pred_norm = normalize(pred_text)

    if gt in {"yes", "no"}:
        return pred_norm == gt or pred_norm.startswith(gt)

    if qtype == "Q3_count":
        return gt in pred_norm.replace(".", " ").split()

    return pred_norm == gt or (gt and gt in pred_norm)


def read_jsonl(path):
    rows = []
    with open(path) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(rows, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--questions", required=True)
    ap.add_argument("--png_orig", required=True)
    ap.add_argument("--crop_root", required=True)
    ap.add_argument("--margins", type=float, nargs="+", default=[1.0, 1.5, 2.0])
    ap.add_argument("--model_id", default="StanfordAIMI/CheXOne")
    ap.add_argument("--out_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    questions = read_jsonl(args.questions)
    if args.limit is not None:
        questions = questions[: args.limit]

    model = build_model(
        "chexone",
        model_id=args.model_id,
        device_map="auto",
        torch_dtype="auto",
        max_new_tokens=32,
    )

    conditions = ["pred_original"] + [f"pred_margin_{m}" for m in args.margins]

    results = []

    for q in tqdm(questions, desc="CheXOne crop VQA"):
        image_id = q["image_id"]
        prompt = build_prompt(q)

        image_paths = {
            "pred_original": str(Path(args.png_orig) / f"{image_id}.png")
        }

        for m in args.margins:
            image_paths[f"pred_margin_{m}"] = str(
                Path(args.crop_root) / f"margin_{m}" / f"{image_id}.png"
            )

        preds = {}
        correct = {}
        errors = {}

        for cond in conditions:
            img_path = image_paths[cond]
            try:
                pred = model.answer(img_path, prompt)
                preds[cond] = pred
                correct[cond] = is_correct(q, pred)
                errors[cond] = None
            except Exception as e:
                preds[cond] = ""
                correct[cond] = None
                errors[cond] = repr(e)

        results.append({
            **q,
            "prompt_used": prompt,
            "preds": preds,
            "correct": correct,
            "errors": errors,
        })

    write_jsonl(results, args.out_jsonl)
    print(f"saved: {args.out_jsonl}")


if __name__ == "__main__":
    main()
