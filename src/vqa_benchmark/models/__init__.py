from vqa_benchmark.models.chexone import CheXOneModel
from vqa_benchmark.models.qwen25vl import Qwen25VLModel


def build_model(name: str, **kwargs):
    name = name.lower()

    if name in {"qwen25vl", "qwen2.5-vl", "qwen2_5_vl"}:
        return Qwen25VLModel(**kwargs)

    if name in {"chexone", "chex-one"}:
        return CheXOneModel(**kwargs)

    raise ValueError(f"Unsupported model: {name}")
