#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-03 14:59
# Filename         : abstract_models.py
# Project          : HuRAG/embedding-service
#*********************************************************************
from abc import ABC, abstractmethod

class AbstractEmbedder(ABC):
    MODEL_NAME_OR_PATH: str
    DEVICE: str
    BATCH_SIZE: int
    NORMALIZE_EMBEDDINGS: bool

    @classmethod
    @abstractmethod
    def encode(
        cls,
        sentences: str|list[str],
        normalize_embeddings: bool|None,
    ):
        pass

    @classmethod
    @abstractmethod
    def startup(
        cls,
        model_name_or_path: str,
        device: str|None,
        batch_size: int|None,
        normalize_embeddings: bool|None,
    ):
        """Initialize, load into cpu/gpu and warm-up model"""
        pass

class AbstractReranker(ABC):
    pass

