from pydantic import BaseModel, Field

class EmbeddingRequest(BaseModel):
    sentences: list[str] | str = Field(
        ...,
        example=["What is LLM", "Something amazing"]
    )
    batch_size: int | None = Field(default=None)
    return_dense: bool = Field(default=True)
    return_sparse: bool = Field(default=False)
    return_colbert_vecs: bool = Field(default=False)
    instruction: str | None = Field(
        default=None,
        example="Embed sentences for retrieval:"
    )

class RerankRequest(BaseModel):
    query: str = Field(..., example="What is LLM")
    documents: list[str] | str = Field(
        ...,
        example=["That is an LLM", "Something amazing", "Large Language Model"]
    )
    query_instruction: str | None = Field(default=None, example="Query:")
    passage_instruction: str | None = Field(default=None, example="Passages:")
    batch_size: int | None = Field(default=None)
    max_length: int | None = Field(default=None)
    normalize: bool | None = Field(default=None)

