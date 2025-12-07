from abc import ABC, abstractmethod

class AbstractEmbedder(ABC):
    def __init__(self):
        raise TypeError(
            "Embedders cannot be instantiated. Use Cls.startup() instead."
        )

    @classmethod
    @abstractmethod
    def encode(
        cls,
        sentences: str | list[str],
        batch_size: int | None,
    ):
        pass

    @classmethod
    @abstractmethod
    def startup(
        cls,
        model_name_or_path: str | None,
        device: str | None,
    ):
        pass

    @classmethod
    @abstractmethod
    def shutdown(cls):
        """Release model resources"""
        pass

class AbstractReranker(ABC):
    def __init__(self):
        raise TypeError(
            "Rerankers cannot be instantiated. Use Cls.startup() instead."
        )

    pass

