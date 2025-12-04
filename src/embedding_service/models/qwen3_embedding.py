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
        sentences: str|list[str],
        batch_size: int|None,
        normalize_embeddings: bool|None,
        **kwargs,
    )-> dict[Literal["dense_vecs", "lexical_weights", "colbert_vecs"], Any]:
        if not sentences:
            return {
                "dense_vecs": None,
                "lexical_weights": None,
                "colbert_vecs":None
            }

        # startup model
        cls.startup(
            model_name_or_path = kwargs.get("model_name_or_path", None),
            device = kwargs.get("device", None),
            batch_size = batch_size,
            normalize_embeddings = normalize_embeddings,
        )
        if cls.model is None:
            return {
                "dense_vecs": None,
                "lexical_weights": None,
                "colbert_vecs":None
            }

        # encoding
        return cls.model.encode(
            [sentences] if not isinstance(sentences, list) else sentences,
            batch_size = batch_size,
        )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str|None=None,
        device: str|None=None,
        batch_size: int|None=None,
        normalize_embeddings: bool|None=None,
    ):
        if cls.model is not None:
            return

        _model_name_or_path = (
            model_name_or_path if model_name_or_path else
            conf.env.model_home + "/" + conf.embedding.qwen3_name
        )
        _device = device or conf.env.device
        _batch_size = batch_size or conf.embedding.batch_size
        _normalize_embeddings = (
            normalize_embeddings
            if normalize_embeddings is not None
            else True
        )

        from sentence_transformers import SentenceTransformer

# --- TODO: breakpoint 1204 ---
        try:
            cls.model = SentenceTransformer(
                _model_name_or_path,
                normalize_embeddings = _normalize_embeddings,
                use_fp16 = False,
                devices = _device,
                batch_size = _batch_size,
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

def ef(
    sentences: str|list[str],
    batch_size: int|None=None,
    normalize_embeddings: bool|None=None,
    **kwargs,
):
    pass

# def ef(
#     sentences: str|list[str],
#     batch_size: int|None=None,
#     normalize_embeddings: bool|None=None,
#     return_dense: bool|None=None,
#     return_sparse: bool|None=None,
#     return_colbert_vecs: bool|None=None,
#     **kwargs,
# )-> dict[Literal["dense_vecs", "lexical_weights", "colbert_vecs"], Any]:
#     """
#     Encode sentences using BGEM3 model. Automatically initializes and warm-up
#     the model if not already done.
# 
#     A single sentence will be converted to a list internally and the resulting
#     embeddings will be in the same format.
# 
#     Args:
#         sentences (str | list[str]):
#             A single sentence or a list of sentences to encode.
#         batch_size (int | None): batch size for encoding. Default is None.
#         normalize_embeddings (bool | None):
#             Whether to normalize the embeddings. Default is None.
#         return_dense (bool | None):
#             Whether to return dense embeddings. Default is None.
#         return_sparse (bool | None):
#             Whether to return sparse embeddings. Default is None.
#         return_colbert_vecs (bool | None):
#             Whether to return ColBERT vectors. Default is None.
#         kwargs: Additional keyword arguments for model configuration includes:
#             model_name_or_path: str, path to the model.
#             device: str, device to run the model on.
#     Returns:
#         dict: A dictionary containing the encoded embeddings.
#     """
#     return BGEM3.encode(
#         sentences,
#         batch_size,
#         normalize_embeddings,
#         return_dense,
#         return_sparse,
#         return_colbert_vecs,
#         **kwargs,
#     )

