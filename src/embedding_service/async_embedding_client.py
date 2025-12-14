import httpx

from .schemas import (
    EmbeddingRequest,
    RerankRequest,
    RerankResponse,
    EmbeddingPayloadMeta,
)
from .utilities import unpack_unified_embeddings_from_bytes


class AsyncEmbeddingClient:
    """Asynchronous client for embedding and reranking services."""

    def __init__(self, base_url: str="http://127.0.0.1:8765"):
        self.base_url = base_url.rstrip("/")
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=10)
        )
        return self

    async def __aexit__(self):
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
        documents: Union[str, List[str]],
        query_instruction: Optional[str] = None,
        passage_instruction: Optional[str] = None,
        batch_size: Optional[int] = None,
        max_length: Optional[int] = None,
        normalize: Optional[bool] = None
    ) -> RerankResponse:
        """对文档进行重排序（JSON响应）"""
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


# ==================== 使用示例 ====================

# async def test_embed():
#     """测试 embedding 接口"""
#     async with EmbeddingClient() as client:
#         embeddings, meta = await client.embed(
#             sentences=["Hello world", "Machine learning is amazing"],
#             return_dense=True,
#             return_sparse=False,
#             return_colbert_vecs=False
#         )
#         
#         print("✅ Embedding 成功")
#         print(f"Meta: {meta}")
#         if embeddings["dense_vecs"] is not None:
#             print(f"Dense shape: {embeddings['dense_vecs'].shape}")
#             print(f"First 5 values: {embeddings['dense_vecs'][0][:5]}")
# 
# 
# async def test_rerank():
#     """测试 rerank 接口"""
#     async with EmbeddingClient() as client:
#         result = await client.rerank(
#             query="What is AI?",
#             documents=[
#                 "Artificial intelligence is the simulation of human intelligence.",
#                 "Machine learning is a subset of AI.",
#                 "Python is a programming language."
#             ],
#             normalize=True
#         )
#         
#         print("\n✅ Rerank 成功")
#         print(f"Scores: {result.scores}")
# 
# 
# async def main():
#     """同时测试两个接口"""
#     await test_embed()
#     await test_rerank()
# 
# 
# if __name__ == "__main__":
#     asyncio.run(main())