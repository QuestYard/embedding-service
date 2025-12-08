from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    sentences: list[str] | str
    batch_size: int | None = None
    return_dense: bool | None = None
    return_sparse: bool | None = None
    return_colbert_vecs: bool | None = None

class RerankRequest(BaseModel):
    query: str
    documents: list[str] | str
