from __future__ import annotations
from typing import TypedDict, TYPE_CHECKING

from .. import logger
from .abstract_models import AbstractEmbedder

if TYPE_CHECKING:
    import numpy

class VectorDict(TypedDict):
    dense_vecs: numpy.ndarray | None

class Qwen3Embedding(AbstractEmbedder):
    _model = None

    @classmethod
    def encode(
        cls,
        sentences: str | list[str],
        batch_size: int = 32,
        instruction: str | None = None,
        **kwargs,
    )-> VectorDict:
        """
        Encode sentences using Qwen3-Embedding-?B model.

        Ensures the model is started up before encoding, otherwise returns
        None for all embeddings.

        A single sentence will be converted to a list internally. Empty input
        will return None for all embeddings.

        Qwen3-Embedding-?B models only provide normalized dense embeddings.

        Args:
            sentences (str | list[str]):
                A single sentence or a list of sentences to encode.
            batch_size (int | None):
                The batch size for encoding. Default is 32.
            instruction (str | None):
                The embed instruction for queries, NOT for documents.

        Returns:
            dict: A dictionary containing the encoded embeddings.
                Let n be the number of sentences, the returned dict will be:
                { "dense_vecs": np.ndarray[(n, 1024), float32] }
        """
        if not sentences:
            logger.warning("No sentences provided for encoding.")
            return { "dense_vecs": None }
        if cls._model is None:
            logger.warning("Model is not started.")
            return { "dense_vecs": None }

        # encoding
        embeddings = cls._model.encode(
            [sentences] if not isinstance(sentences, list) else sentences,
            batch_size=batch_size,
            prompt=instruction,
            convert_to_numpy=True,
            convert_to_tensor=False,
        )
        return { "dense_vecs": embeddings }

    @classmethod
    def startup(
        cls,
        model_name_or_path: str,
        device: str | None = None,
        **kwargs,
    )-> None:
        """
        Initialize the Qwen3-Embedding model if not already initialized.
        
        Args:
            model_name_or_path (str):
                Path to the model. None or empty will cause a ValueError.
            device (str | None):
                Device to run the model on.
        """
        if cls._model is not None:
            return

        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided.")

        from sentence_transformers import SentenceTransformer

        try:
            cls._model = SentenceTransformer(
                model_name_or_path.strip(),
                device = device,
            )
            logger.info(f"{model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {model_name_or_path} failed: {e}")
            cls._model = None
            return

        try:
            _ = cls._model.encode("hello")
            logger.info(f"{model_name_or_path} warmed-up")
        except Exception as e:
            logger.error(f"Warming-up {model_name_or_path} failed: {e}")
            cls._model = None
            return

    @classmethod
    def shutdown(cls)-> None:
        if cls._model is not None:
            cls._model = None
            logger.info("Qwen3Embedding model shutdown.")
