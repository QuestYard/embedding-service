import httpx

from .schemas import (
    EmbeddingRequest,
    RerankRequest,
    RerankResponse,
    EmbeddingPayloadMeta,
)
from .adapters import unpack_unified_embeddings_from_bytes


class AsyncEmbeddingClient:
    """Asynchronous client for embedding and reranking services."""

    def __init__(self, base_url: str, timeout: float=300.0):
        self.base_url = base_url.strip().rstrip("/")
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=10)
        )
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self._client:
            await self._client.aclose()

    def _ensure_client(self):
        if not self._client:
            raise RuntimeError(
                "Client not initialized. Use 'async with' context manager."
            )

    async def embed(
        self,
        sentences: str | list[str],
        batch_size: int | None = None,
        return_dense: bool = True,
        return_sparse: bool = False,
        return_colbert_vecs: bool = False,
        instruction: str | None = None
    ) -> tuple[dict, EmbeddingPayloadMeta]:
        """
        Get embeddings for the given sentences (unified format with metadata).

        Args:
            sentences (str | list[str]):
                A single sentence or a list of sentences to encode.
            batch_size (int | None):
                The batch size for encoding.
            return_dense (bool):
                Whether to return dense embeddings.
            return_sparse (bool):
                Whether to return sparse embeddings.
            return_colbert_vecs (bool):
                Whether to return ColBERT vectors.
            instruction (str | None):
                The embed instruction for queries, NOT for documents.

        Returns:
            tuple[dict, EmbeddingPayloadMeta]:
                A tuple containing:
                - A dictionary with the encoded embeddings.
                - An EmbeddingPayloadMeta object with metadata.
        """
        self._ensure_client()

        request = EmbeddingRequest(
            sentences=sentences,
            batch_size=batch_size,
            return_dense=return_dense,
            return_sparse=return_sparse,
            return_colbert_vecs=return_colbert_vecs,
            instruction=instruction
        )

        async with self._client.stream(
            "POST",
            f"{self.base_url}/embed",
            json=request.model_dump()
        ) as response:
            response.raise_for_status()

            chunks = []
            async for chunk in response.aiter_bytes(chunk_size=8192):
                chunks.append(chunk)

            packed_bytes = b"".join(chunks)

        embd, meta = unpack_unified_embeddings_from_bytes(packed_bytes)
        return embd, meta

    async def rerank(
        self,
        query: str,
        documents: str | list[str],
        query_instruction: str | None = None,
        passage_instruction: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        normalize: bool | None = None
    ) -> RerankResponse:
        """
        Rerank documents based on their relevance to the query.

        Args:
            query (str):
                The query string to compare against documents.
            documents (str | list[str]):
                A single document or a list of documents to be reranked.
            query_instruction (str | None):
                Instruction for queries.
            passage_instruction (str | None):
                Instruction for passages.
            batch_size (int | None):
                The batch size for processing documents.
            max_length (int | None):
                The max length of context.
            normalize (bool | None):
                Whether to normalize the scores.

        Returns:
            RerankResponse:
                An RerankResponse object containing the reranked scores.
        """
        self._ensure_client()
        
        request = RerankRequest(
            query=query,
            documents=documents,
            query_instruction=query_instruction,
            passage_instruction=passage_instruction,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize
        )
        
        response = await self._client.post(
            f"{self.base_url}/rerank",
            json=request.model_dump()
        )
        response.raise_for_status()
        
        return RerankResponse.model_validate(response.json())

