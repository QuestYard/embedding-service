#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-04 20:50
# Filename         : models/qwen3_embedding.py
# Project          : HuRAG/embedding-service
#*********************************************************************
from .. import conf, logger
from .abstract_models import AbstractEmbedder

from typing import Literal, Any

class Qwen3Embedding(AbstractEmbedder):
    model = None

    @classmethod
    def encode(
        cls,
    ):
        pass

    @classmethod
    def startup(
        cls,
    ):
        pass

def ef(
    sentences: str|list[str],
    batch_size: int|None=None,
    normalize_embeddings: bool|None=None,
    **kwargs,
):
    pass
