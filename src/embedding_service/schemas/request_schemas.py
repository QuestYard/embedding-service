from pydantic import BaseModel

class EmbeddingRequest(BaseModel):
    sentences: list[str] | str
    batch_size: int | None = None
    return_dense: bool = True
    return_sparse: bool = False
    return_colbert_vecs: bool = False
    instruction: str | None = None

class RerankRequest(BaseModel):
    query: str
    documents: list[str] | str
    query_instruction: str | None = None
    passage_instruction: str | None = None
    batch_size: int | None = None
    max_length: int | None = None
    normalize: bool | None = None

