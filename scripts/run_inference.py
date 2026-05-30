import argparse
import tempfile
from pathlib import Path

from tqdm import tqdm

from vqa_benchmark.data.vindr_loader import save_dicom_as_png
from vqa_benchmark.models import build_model
from vqa_benchmark.utils.io import load_yaml, read_jsonl, write_jsonl
from vqa_benchmark.utils.seed import set_seed


def maybe_convert_dicom(image_path: Path, tmp_dir: Path) -> Path:
    if image_path.suffix.lower() in {".dicom", ".dcm"}:
        output_path = tmp_dir / f"{image_path.stem}.png"
        save_dicom_as_png(str(image_path), str(output_path))
        return output_path

    return image_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--qa", type=str, required=True)
    parser.add_argument("--image-root", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--model-id", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(cfg["project"].get("seed", 42))

    model_name = args.model_name or cfg["model"]["name"]
    model_id = args.model_id or cfg["model"]["hf_id"]
    image_root = Path(args.image_root or cfg["data"]["image_root"])

    model = build_model(
        model_name,
        model_id=model_id,
        device_map=cfg["model"].get("device_map", "auto"),
        torch_dtype=cfg["model"].get("torch_dtype", "auto"),
        max_new_tokens=cfg["model"].get("max_new_tokens", 32),
    )

    qa_items = read_jsonl(args.qa)
    results = []

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)

        for item in tqdm(qa_items, desc="Running VQA"):
            image_path = Path(item["image_path"])

            if not image_path.is_absolute():
                image_path = image_root / image_path

            question = item["question"]
            prompt_suffix = cfg["inference"].get("prompt_suffix", "")

            if prompt_suffix:
                question = f"{question} {prompt_suffix}"

            try:
                model_image_path = maybe_convert_dicom(image_path, tmp_dir)
                pred = model.answer(str(model_image_path), question)
                error = None
            except Exception as e:
                pred = ""
                error = repr(e)

            results.append(
                {
                    **item,
                    "prediction": pred,
                    "model_name": model_name,
                    "model_id": model_id,
                    "image_used": str(image_path),
                    "error": error,
                }
            )

    write_jsonl(results, args.output)
    print(f"Saved predictions to {args.output}")


if __name__ == "__main__":
    main()
