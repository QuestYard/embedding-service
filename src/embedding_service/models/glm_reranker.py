from .. import logger
from .abstract_models import AbstractReranker

import os
import httpx

class GLMReranker(AbstractReranker):
    _model = None   # an Httpx.AsyncClient
    _url = None
    _api_key = None
    _header = None

    @classmethod
    def rank(
        cls,
        query: str,
        passages: str | list[str],
        query_instruction: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        **kwargs,
    )-> list[float]:
        """
        Rerank passages based on their relevance to the query using BGE model.

        Ensures the model is started up before reranking, otherwise returns
        an empty list.

        Qwen3-Reranker models confirm to normalize the final scores. It does
        not use passage instruction for reranker.

        Args:
            query (str):
                The query string to compare against passages.
            passages (list[str]):
                A list of passage strings to be reranked.
            query_instruction: (str | None):
                Instruction for queries.
            batch_size (int | None):
                The batch size for processing passages. Default is None.
            max_length (int | None):
                The max length of context, default 3072.

        Returns:
            list[float]: A list of relevance scores for each passage.
        """
        ...
#         if not query:
#             logger.warning("Query must be provided.")
#             return []
#         if not passages:
#             logger.warning("Passages must be provided.")
#             return []
#         if cls._model is None:
#             logger.warning("Model is not started.")
#             return []
# 
#         pairs = [[query, passages]] if isinstance(passages, str) else [
#             [query, p] for p in passages
#         ]
# 
#         return cls._model.compute_score(
#             pairs,
#             instruction=query_instruction,
#             batch_size=batch_size,
#             max_length=max_length,
#         )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str,
        glm_api_key: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 128,
        **kwargs,
    )-> None:
        """
        Initialize an Httpx.AsyncClient object for GLM Rerank model.

        Args:
            model_name_or_path (str): url
            glm_api_key (str | None): GLM Token
            device (str | None): not used
            query_instruction: (str | None): not used
            batch_size: (int): not used
        """
        if cls._model is not None:
            return

        if not model_name_or_path:
            raise ValueError("url must be provided.")
        else:
            cls._url = model_name_or_path

        if not glm_api_key:
            raise ValueError("GLM token must be provided.")
        else:
            cls._api_key = glm_api_key
# --- to be continued ---
#         try:
#             cls._model = Qwen3RerankerModel(
#                 model_name_or_path.strip(),
#                 device = device,
#                 instruction = query_instruction,
#                 batch_size = batch_size,
#             )
#             logger.info(f"{model_name_or_path} loaded.")
#         except Exception as e:
#             logger.error(f"Loading {model_name_or_path} failed: {e}")
#             cls._model = None
#             return
# 
#         try:
#             _ = cls._model.compute_score([('query', 'passage')])
#             logger.info(f"{model_name_or_path} warmed-up.")
#         except Exception as e:
#             logger.error(f"Warming-up {model_name_or_path} failed: {e}")
#             return

    @classmethod
    def shutdown(cls):
        ...
#         if cls._model is None:
#             return
# 
#         try:
#             cls._model.shutdown()
#             logger.info("Qwen3-Reranker model shutdown.")
#         except:
#             pass
#         finally:
#             cls._model = None


