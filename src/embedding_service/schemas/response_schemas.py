from pydantic import BaseModel


class CSRMeta(BaseModel):
    nnz: int
    shape: tuple[int, int]
    dtype: str


class ColBertMeta(BaseModel):
    count: int
    shapes: list[tuple[int, ...]]
    dtype: str


class EmbeddingPayloadMeta(BaseModel):
    has_dense: bool
    dense_shape: tuple[int, int] | None = None
    dense_dtype: str | None = None

    has_sparse: bool
    sparse_meta: CSRMeta | None = None

    has_colbert: bool
    colbert_meta: ColBertMeta | None = None

    format_version: str = "npz_v1"


class RerankResponse(BaseModel):
    scores: list[float] | None = None
