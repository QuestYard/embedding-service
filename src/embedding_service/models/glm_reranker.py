from .. import logger
from .abstract_models import AbstractReranker

import os
import threading
import httpx
import asyncio
from typing import Any

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception,
)

def should_retry_error(e):
    # Retry on 429 Rate Limit
    if isinstance(e, httpx.HTTPStatusError) and e.response.status_code == 429:
        return True
    # Retry on connection errors or timeouts (transient network issues)
    if isinstance(
        e, (httpx.ConnectError, httpx.TimeoutException, httpx.ReadTimeout)
    ):
        return True
    return False

class GLMReranker(AbstractReranker):
    _client = None  # an Httpx.AsyncClient
    _loop = None
    _thread = None
    _model = None
    _base_url = None
    _api_key = None
    _headers = None
    _lock = threading.Lock()

    @staticmethod
    def _start_background_loop(loop: asyncio.AbstractEventLoop) -> None:
        asyncio.set_event_loop(loop)
        loop.run_forever()

    @classmethod
    def rank(
        cls,
        query: str,
        passages: str | list[str],
        batch_size: int = 2,
        query_instruction: str | None = None,
        max_length: int | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Rerank passages based on their relevance to the query using GLM rerank model.

        Ensures the model is started up before reranking, otherwise returns
        an empty list.

        GLM rerank confirm to return raw scores.

        Args:
            query (str):
                The query string to compare against passages.
            passages (list[str]):
                A list of passage strings to be reranked.
            query_instruction: (str | None): not used.
            batch_size (int):
                The batch size for processing passages. Default is 2.
            max_length (int | None): not used.

        Returns:
            list[float]: A list of relevance scores for each passage.
        """
        if not query:
            logger.warning("Query must be provided.")
            return []
        if not passages:
            logger.warning("Passages must be provided.")
            return []
        if cls._client is None:
            logger.warning("Model is not started.")
            return []

        if isinstance(passages, str):
            passages = [passages]

        with cls._lock:
            future = asyncio.run_coroutine_threadsafe(
                parallel_glm_rerank(
                    query=query,
                    documents=passages,
                    model=cls._model,
                    base_url=cls._base_url,
                    client=cls._client,
                    batch_size=batch_size,
                    workers=20,
                ),
                cls._loop,
            )
        return future.result()

    @classmethod
    def startup(
        cls,
        model_name_or_path: str,
        glm_model: str | None = None,
        glm_api_key: str | None = None,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 2,
        **kwargs,
    ) -> None:
        """
        Initialize an Httpx.AsyncClient object for GLM Rerank model.

        Args:
            model_name_or_path (str): GLM rerank URL
            glm_model (str | None): GLM model name
            glm_api_key (str | None): GLM Token
            device (str | None): not used
            query_instruction: (str | None): not used
            batch_size: (int): not used
        """
        if cls._client is not None:
            return

        if not model_name_or_path:
            raise ValueError("base_url must be provided.")
        else:
            cls._base_url = model_name_or_path

        if not glm_model:
            raise ValueError("GLM model must be provided.")
        else:
            cls._model = glm_model

        if not glm_api_key:
            raise ValueError("GLM token must be provided.")
        else:
            cls._api_key = glm_api_key

        # Start background loop
        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()
            cls._thread = threading.Thread(
                target=cls._start_background_loop,
                args=(cls._loop,),
                daemon=True
            )
            cls._thread.start()

        cls._headers = {
            "Authorization": f"Bearer {cls._api_key}",
            "Content-Type": "application/json",
        }
        try:
            cls._client = httpx.AsyncClient(
                headers=cls._headers,
                timeout = 60.0,
                limits = httpx.Limits(
                    max_connections=50,
                    max_keepalive_connections=30,
                    keepalive_expiry=60.0,
                ),
            )
            logger.info(f"{model_name_or_path} client created.")
        except Exception as e:
            logger.error(f"{model_name_or_path} client creating failed: {e!r}")
            cls._client = None
            cls._model = None
            cls._base_url = None
            cls._api_key = None
            cls._headers = None
            return

        try:
            future = asyncio.run_coroutine_threadsafe(
                glm_rerank(
                    query="query",
                    documents=["passage1", "passage2"],
                    model=cls._model,
                    base_url=cls._base_url,
                    client=cls._client,
                ),
                cls._loop,
            )
            future.result()
            logger.info(f"{model_name_or_path} warmed-up.")
        except Exception as e:
            logger.error(f"Warming-up {model_name_or_path} failed: {e!r}")
            return

    @classmethod
    def shutdown(cls):
        if cls._client is None:
            return

        try:
            if cls._loop and cls._loop.is_running():
                future = asyncio.run_coroutine_threadsafe(
                    glm_rerank_close(cls._client),
                    cls._loop
                )
                future.result()
            logger.info("GLM-Rerank client closed.")
        finally:
            cls._client = None
            cls._model = None
            cls._base_url = None
            cls._api_key = None
            cls._headers = None

            if cls._loop and cls._loop.is_running():
                cls._loop.call_soon_threadsafe(cls._loop.stop)
                if cls._thread:
                    cls._thread.join()
            cls._loop = None
            cls._thread = None

async def glm_rerank_close(client: httpx.AsyncClient):
    await client.aclose()
        
@retry(
    retry=retry_if_exception(should_retry_error),
    stop=stop_after_attempt(4),
    wait=wait_exponential(multiplier=1, max=10),
)
async def glm_rerank(
    query: str,
    documents: list[str],
    model: str,
    base_url: str,
    client: httpx.AsyncClient,
) -> list[dict[str, Any]]:
    """
    Rerank documents using GLM Reranker.

    Returns:
        list[dict[str, Any]] -- list of reranked documents with scores.
            Each dict contains 'index' and 'relevance_score' keys.
    """
    payload = {
        "model": model,
        "query": query,
        "documents": documents,
        "return_raw_scores": True,
    }
    response = await client.post(
        base_url,
        json=payload,
        timeout=30.0,
    )
    response.raise_for_status()
    result = response.json()
    return result["results"]

async def parallel_glm_rerank(
    query: str,
    documents: list[str],
    model: str,
    base_url: str,
    client: httpx.AsyncClient,
    batch_size: int = 2,
    workers: int = 20,
) -> list[float]:
    """
    Rerank documents in parallel using GLM Reranker.

    Args:
        query: str -- the query string.
        documents: list[str] -- list of document strings to rerank.
        model: the model name of GLM rerank.
        base_url: the base url of GLM rerank.
        client: httpx.AsyncClient.
        batch_size: int -- number of documents per batch.
        workers: int -- number of parallel workers.

    Returns:
        list[float] -- list of relevance scores corresponding to documents.
    """
    async def _worker(queue: asyncio.Queue):

        while True:
            batch_indices = await queue.get()
            batch_docs = [documents[i] for i in batch_indices]
            try:
                reranked = await glm_rerank(
                    query,
                    batch_docs,
                    model=model,
                    base_url=base_url,
                    client=client,
                )
                for score in reranked:
                    scores[batch_indices[score["index"]]] = score["relevance_score"]
            except Exception as e:
                logger.error(f"Failed to rerank batch using GLM: {e!r}")
            finally:
                queue.task_done()

    queue = asyncio.Queue()
    scores = [0.0] * len(documents)
    rerankers = [asyncio.create_task(_worker(queue)) for _ in range(workers)]

    for i in range(0, len(documents), batch_size):
        batch_indices = list(range(i, min(i + batch_size, len(documents))))
        await queue.put(batch_indices)

    await queue.join()
    for worker in rerankers:
        worker.cancel()
    gathered = await asyncio.gather(*rerankers, return_exceptions=True)

    return scores

