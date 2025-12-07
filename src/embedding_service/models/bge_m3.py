from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING

from .. import conf, logger
from .abstract_models import AbstractEmbedder

if TYPE_CHECKING:
    import numpy

class VectorDict(TypedDict):
    dense_vecs: numpy.ndarray | None
    lexical_weights: list[dict[str, float]] | None
    colbert_vecs: list[numpy.ndarray] | None

class BGEM3(AbstractEmbedder):
    _model = None

    @classmethod
    def encode(
        cls,
        sentences: str | list[str],
        batch_size: int | None=None,
        return_dense: bool | None=None,
        return_sparse: bool | None=None,
        return_colbert_vecs: bool | None=None,
    )-> VectorDict:
        """
        Encode sentences using BGEM3 model. Ensures the model is started up
        before encoding, otherwise returns None for all embeddings.

        A single sentence will be converted to a list internally. Empty input
        will return None for all embeddings.

        Args:
            sentences (str | list[str]):
                A single sentence or a list of sentences to encode.
            batch_size (int | None):
                The batch size for encoding. Default is None, leading to use the
                batch size set during startup.
            return_dense (bool | None):
                Whether to return dense embeddings. Default is None.
            return_sparse (bool | None):
                Whether to return sparse embeddings. Default is None.
            return_colbert_vecs (bool | None):
                Whether to return ColBERT vectors. Default is None.

        Returns:
            dict: A dictionary containing the encoded embeddings.
                Let n be the number of sentences, the returned dict will be:
                {
                    "dense_vecs": np.ndarray[(n, 1024), float32],
                    "lexical_weights": list[defaultdict[str, float32], len=n],
                    "colbert_vecs": list[np.ndarray[(x, 1024), float32], len=n],
                }
        """
        if not sentences:
            logger.warning("No sentences provided for encoding.")
            return {
                "dense_vecs": None,
                "lexical_weights": None,
                "colbert_vecs":None
            }
        if cls._model is None:
            logger.warning("Model is not started.")
            return {
                "dense_vecs": None,
                "lexical_weights": None,
                "colbert_vecs":None
            }
        # encoding
        return cls._model.encode(
            [sentences] if not isinstance(sentences, list) else sentences,
            batch_size = batch_size,
            return_dense = return_dense,
            return_sparse = return_sparse,
            return_colbert_vecs = return_colbert_vecs,
        )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str | None=None,
        device: str | None=None,
        batch_size: int=16,
        normalize_embeddings: bool=True,
        return_dense: bool=True,
        return_sparse: bool=False,
        return_colbert_vecs: bool=False,
    )-> None:
        """
        Initialize, load and warm-up BGEM3 model.

        The batch size and vector return options are set during startup and
        will be used as defaults during encoding unless overridden in the
        encode call.

        The normalization option is set during startup and cannot be changed
        during encoding. Changes to normalization require a model restart.

        It is strongly recommended to always normalize embeddings unless there
        is a specific need for unnormalized embeddings.

        Args:
            model_name_or_path (str | None):
                Path to the model. If None, use default from config.
            device (str | None):
                Device to run the model on. If None, use default from config.
            batch_size (int): batch size for encoding. Default is 16.
            normalize_embeddings (bool):
                Whether to normalize the embeddings. Default is True.
            return_dense (bool):
                Whether to return dense embeddings. Default is True.
            return_sparse (bool):
                Whether to return sparse embeddings. Default is False.
            return_colbert_vecs (bool):
                Whether to return ColBERT vectors. Default is False.
        """
        if cls._model is not None:
            return

        _model_name_or_path = (
            model_name_or_path if model_name_or_path else
            conf.env.model_home + "/" + conf.embedding.bge_name
        )
        if not return_sparse and not return_colbert_vecs:
            return_dense = True

        from FlagEmbedding import BGEM3FlagModel

        try:
            cls._model = BGEM3FlagModel(
                _model_name_or_path,
                normalize_embeddings = normalize_embeddings,
                use_fp16 = False,
                devices = device or conf.env.device,
                batch_size = batch_size or conf.embedding.batch_size or 16,
                return_dense = return_dense,
                return_sparse = return_sparse,
                return_colbert_vecs = return_colbert_vecs,
            )
            logger.info(f"{_model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {_model_name_or_path} failed: {e}")
            cls._model = None
            return

        try:
            _ = cls._model.encode("hello")
            logger.info(f"{_model_name_or_path} warmed-up")
        except Exception as e:
            logger.error(f"Warming-up {_model_name_or_path} failed: {e}")
            cls._model = None
            return

    @classmethod
    def shutdown(cls)-> None:
        if cls._model is not None:
            cls._model = None
            logger.info("BGEM3 model shutdown.")
