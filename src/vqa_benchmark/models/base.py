from abc import ABC, abstractmethod


class VQAModel(ABC):
    @abstractmethod
    def answer(self, image_path: str, question: str) -> str:
        raise NotImplementedError
