#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-06 23:36
# Filename         : qwen3_embedding.py
# Project          : HuRAG/embedding-service
#*********************************************************************
from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING

from .. import conf, logger
from .abstract_models import AbstractEmbedder

if TYPE_CHECKING:
    import numpy

class VectorDict(TypedDict):
    dense_vecs: numpy.ndarray | None

class Qwen3Embedding(AbstractEmbedder):
    model = None

    @classmethod
    def encode(
        cls,
        sentences: str|list[str],
        batch_size: int|None,
        **kwargs,
    )-> VectorDict:
        if not sentences:
            return { "dense_vecs": None }

        # startup model
        cls.startup(
            model_name_or_path = kwargs.get("model_name_or_path", None),
            device = kwargs.get("device", None),
        )
        if cls.model is None:
            return { "dense_vecs": None }

        # encoding
        _batch_size = batch_size or conf.embedding.batch_size
        return cls.model.encode(
            [sentences] if not isinstance(sentences, list) else sentences,
            batch_size = _batch_size,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str|None=None,
        device: str|None=None,
    ):
        if cls.model is not None:
            return

        _model_name_or_path = (
            model_name_or_path if model_name_or_path else
            conf.env.model_home + "/" + conf.embedding.qwen3_name
        )
        _device = device or conf.env.device

        from sentence_transformers import SentenceTransformer

        try:
            cls.model = SentenceTransformer(
                _model_name_or_path,
                device = _device,
            )
            logger.info(f"{_model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {_model_name_or_path} failed: {e}")
            cls.model = None
            return

        try:
            _ = cls.model.encode("hello")
            logger.info(f"{_model_name_or_path} warmed-up")
        except Exception as e:
            logger.error(f"Warming-up {_model_name_or_path} failed: {e}")
            cls.model = None
            return

    @classmethod
    def shutdown(cls):
        if cls.model is not None:
            cls.model = None
            logger.info("Qwen3Embedding model shutdown.")

def ef(
    sentences: str|list[str],
    batch_size: int|None=None,
    **kwargs,
)-> VectorDict:
    """
    Encode sentences using Qwen3-Embedding model. Automatically initializes and
    warm-up the model if not already done.

    A single sentence will be converted to a list internally and the resulting
    embeddings will be in the same format.

    Qwen3-Embedding models are restricted to output normalized embeddings.

    Args:
        sentences (str | list[str]):
            A single sentence or a list of sentences to encode.
        batch_size (int | None): batch size for encoding. Default is None.
        kwargs: Additional keyword arguments for model configuration includes:
            model_name_or_path: str, path to the model.
            device: str, device to run the model on.
    Returns:
        dict: A dictionary containing the encoded embeddings.
            Let n be the number of sentences, the returned dict will be:
            { "dense_vecs": np.ndarray[(n, 1024), float32] }
    """
    return Qwen3Embedding.encode(
        sentences,
        batch_size,
        **kwargs,
    )

def shutdown():
    """
    Shutdown the Qwen3Embedding model to free up resources.
    """
    Qwen3Embedding.shutdown()
