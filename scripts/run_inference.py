import argparse
import random
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm

from vqa_benchmark.data.vindr_loader import (
    ensure_png_from_dicom,
    resolve_cached_png_path,
    resolve_vindr_dicom_path,
)
from vqa_benchmark.models import build_model
from vqa_benchmark.utils.io import load_yaml, read_jsonl, write_jsonl
from vqa_benchmark.utils.seed import set_seed


def get_record_id(item: Dict) -> str:
    if "qa_id" in item:
        return str(item["qa_id"])
    if "qid" in item:
        return str(item["qid"])
    raise KeyError("QA item must contain either 'qa_id' or 'qid'.")


def get_qtype(item: Dict) -> str:
    return item.get("question_type") or item.get("qtype") or "unknown"


def build_image_path(
    item: Dict,
    image_root: Path,
    png_root: Path,
    long_side: int = 1024,
) -> str:
    """
    Supports both old schema:
      image_path: test/xxx.dicom

    and new benchmark schema:
      image_id: xxx
    """
    split = item.get("split", "test")

    if "image_path" in item:
        raw_path = Path(item["image_path"])
        if raw_path.is_absolute():
            dicom_path = raw_path
        else:
            dicom_path = image_root / raw_path

        image_id = item.get("image_id", dicom_path.stem)
    else:
        image_id = item["image_id"]
        dicom_path = resolve_vindr_dicom_path(
            image_root=str(image_root),
            image_id=image_id,
            split=split,
        )

    png_path = resolve_cached_png_path(
        png_root=str(png_root),
        image_id=image_id,
        split=split,
    )

    return ensure_png_from_dicom(
        dicom_path=str(dicom_path),
        png_path=str(png_path),
        long_side=long_side,
    )


def format_options(options: List[str], shuffle: bool, seed: int, qid: str) -> List[str]:
    options = [str(o) for o in options]

    if not shuffle:
        return options

    rng = random.Random(f"{seed}_{qid}")
    shuffled = list(options)
    rng.shuffle(shuffled)
    return shuffled


def build_prompt(
    item: Dict,
    prompt_suffix: str = "",
    shuffle_options: bool = False,
    option_seed: int = 42,
) -> tuple[str, Optional[List[str]]]:
    question = item["question"]
    qid = get_record_id(item)

    used_options = None

    if "options" in item and item["options"] is not None:
        used_options = format_options(
            options=item["options"],
            shuffle=shuffle_options,
            seed=option_seed,
            qid=qid,
        )

        option_lines = "\n".join(
            [f"{chr(ord('A') + i)}. {opt}" for i, opt in enumerate(used_options)]
        )

        prompt = (
            f"{question}\n\n"
            f"Options:\n"
            f"{option_lines}\n\n"
            f"Answer with only the option text."
        )
    else:
        prompt = question

    if prompt_suffix:
        prompt = f"{prompt}\n\n{prompt_suffix}"

    return prompt, used_options


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--png-root", type=str, default="data/processed/png")
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shuffle-options", action="store_true")
    parser.add_argument("--option-seed", type=int, default=42)
    parser.add_argument("--long-side", type=int, default=1024)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    seed = cfg["project"].get("seed", 42)
    set_seed(seed)

    model_name = args.model_name or cfg["model"]["name"]
    model_id = args.model_id or cfg["model"]["hf_id"]
    image_root = Path(args.image_root or cfg["data"]["image_root"])
    png_root = Path(args.png_root)

    model = build_model(
        model_name,
        model_id=model_id,
        device_map=cfg["model"].get("device_map", "auto"),
        torch_dtype=cfg["model"].get("torch_dtype", "auto"),
        max_new_tokens=cfg["model"].get("max_new_tokens", 32),
    )

    qa_items = read_jsonl(args.qa)

    if args.limit is not None:
        qa_items = qa_items[: args.limit]

    results = []

    for item in tqdm(qa_items, desc="Running VQA"):
        qid = get_record_id(item)
        qtype = get_qtype(item)

        try:
            model_image_path = build_image_path(
                item=item,
                image_root=image_root,
                png_root=png_root,
                long_side=args.long_side,
            )

            prompt, used_options = build_prompt(
                item=item,
                prompt_suffix=cfg["inference"].get("prompt_suffix", ""),
                shuffle_options=args.shuffle_options,
                option_seed=args.option_seed,
            )

            pred = model.answer(model_image_path, prompt)
            error = None

        except Exception as e:
            model_image_path = ""
            prompt = ""
            used_options = None
            pred = ""
            error = repr(e)

        results.append(
            {
                **item,
                "qa_id": qid,
                "question_type": qtype,
                "prediction": pred,
                "model_name": model_name,
                "model_id": model_id,
                "image_used": model_image_path,
                "prompt_used": prompt,
                "options_used": used_options,
                "shuffle_options": args.shuffle_options,
                "error": error,
            }
        )

    write_jsonl(results, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
