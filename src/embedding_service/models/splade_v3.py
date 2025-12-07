from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING

from .. import conf, logger
from .abstract_models import AbstractEmbedder

if TYPE_CHECKING:
    from torch import Tensor

class VectorDict(TypedDict):
    lexical_weights: Tensor | None

class Splade_v3(AbstractEmbedder):
    _model = None

    @classmethod
    def encode(
        cls,
        sentences: str | list[str],
        batch_size: int | None=None,
    )-> VectorDict:
        """
        Encode sentences using Splade_v3 model. Ensures the model is started up
        before encoding, otherwise returns None for all embeddings.

        A single sentence will be converted to a list internally. Empty input
        will return None for all embeddings.

        Args:
            sentences (str | list[str]):
                A single sentence or a list of sentences to encode.
            batch_size (int | None):
                The batch size for encoding. Default is None, leading to use the
                batch size set during startup.

        Returns:
            dict: A dictionary containing the encoded embeddings.
                Let n be the number of sentences, the returned dict will be:
                {
                    "lexical_weights": Tensor[(n, 30522), sparse_coo],
                }
        """
        if not sentences:
            logger.warning("No sentences provided for encoding.")
            return {
                "lexical_weights": None,
            }
        if cls._model is None:
            logger.error("Model is not started.")
            return {
                "lexical_weights": None,
            }
        # encoding
        lexical_weights = cls._model.encode(
            [sentences] if not isinstance(sentences, list) else sentences,
            batch_size=batch_size or conf.embedding.batch_size or 16,
        )
        return {
            "lexical_weights": lexical_weights,
        }

    @classmethod
    def startup(
        cls,
        model_name_or_path: str | None=None,
        device: str | None=None,
    )-> None:
        """
        Startup the Splade_v3 model.

        Args:
            model_name_or_path (str | None):
                The model name or path. Default is None, leading to use the
                model name set in configuration.
            device (str | None):
                Device to run the model on. If None, uses default from config.
        """
        if cls._model is not None:
            return

        _model_name_or_path = (
            model_name_or_path if model_name_or_path else
            conf.env.model_home + "/" + conf.embedding.splade_name
        )

        from sentence_transformers import SparseEncoder

        try:
            cls._model = SparseEncoder(
                _model_name_or_path,
                device = device or conf.env.device,
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
            logger.info("Splade_v3 model shutdown.")
