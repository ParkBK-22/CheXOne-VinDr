from vqa_benchmark.models.qwen25vl import Qwen25VLModel


class CheXOneModel(Qwen25VLModel):
    def __init__(self, **kwargs):
        kwargs.setdefault("model_id", "StanfordAIMI/CheXOne")
        super().__init__(**kwargs)
