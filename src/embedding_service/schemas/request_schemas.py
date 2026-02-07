from pydantic import BaseModel, Field


class EmbeddingRequest(BaseModel):
    sentences: list[str] | str = Field(examples=[["What is LLM", "Something amazing"]])
    batch_size: int | None = Field(default=None, examples=[16])
    return_dense: bool = Field(default=True)
    return_sparse: bool = Field(default=False)
    return_colbert_vecs: bool = Field(default=False)
    instruction: str | None = Field(
        default=None, examples=["Embed sentences for retrieval:"]
    )


class RerankRequest(BaseModel):
    query: str = Field(examples=["What is LLM"])
    documents: list[str] | str = Field(
        examples=[["That is an LLM", "Something amazing", "Large Language Model"]]
    )
    query_instruction: str | None = Field(default=None, examples=["Query:"])
    passage_instruction: str | None = Field(default=None, examples=["Passages:"])
    batch_size: int | None = Field(default=None, examples=[4])
    max_length: int | None = Field(default=None, examples=[2048])
    normalize: bool | None = Field(default=None, examples=[True])
