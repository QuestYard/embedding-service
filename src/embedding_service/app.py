from __future__ import annotations
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

from . import logger, conf, __version__ as app_version
from .schemas import EmbeddingRequest, RerankRequest, RerankResponse
from .adapters import unify_embeddings, pack_unified_embeddings_to_bytes
from .models import BGEM3, Qwen3Embedding, Splade_v3
from .models import BGEReranker, Qwen3Reranker

# Model registry
EMBEDDING_MODELS = {
    "bge": BGEM3,
    "qwen3": Qwen3Embedding,
}

SPARSE_MODELS = {
    "bge": BGEM3,
    "splade": Splade_v3,
}

RERANKER_MODELS = {
    "bge": BGEReranker,
    "qwen3": Qwen3Reranker,
}

class ModelManager:
    """Manages model lifecycle and configuration."""

    def __init__(self):
        self.dense_model = None
        self.sparse_model = None
        self.reranker_model = None
        self.sparse_type = None
        self.dense_type = None

    def startup_models(self)-> None:
        if not conf:
            raise RuntimeError("Configuration not loaded")

        # Get device and model home from config
        model_home = conf.env.model_home.rstrip("/")

        # Initialize dense model
        dense_model_type = conf.embedding.dense_model
        if dense_model_type not in EMBEDDING_MODELS:
            logger.warning("Unknown embedding model type, using bge-m3.")
            dense_model_type = "bge"

        self.dense_type = dense_model_type

        model_name = (
            conf.embedding.bge_name if dense_model_type == "bge"
            else conf.embedding.qwen3_name
        )
        if not model_name:
            raise ValueError(
                f"{dense_model_type}_name must be specified in config"
            )

        if model_home:
            model_path = f"{model_home}/{model_name}"
        else:
            model_path = model_name

        logger.info(f"Starting up model: {model_path}")
        self.dense_model = EMBEDDING_MODELS[dense_model_type]
        self.dense_model.startup(model_path, device=conf.env.device)

        # Initialize sparse model (if different from dense model)
        sparse_model_type = conf.embedding.sparse_model
        if sparse_model_type not in SPARSE_MODELS:
            logger.warning("Unknown sparse model type, using bge-m3.")
            sparse_model_type = "bge"

        self.sparse_type = sparse_model_type

        if sparse_model_type == dense_model_type == "bge":
            self.sparse_model = self.dense_model
        else:
            model_name = (
                conf.embedding.bge_name if sparse_model_type == "bge"
                else conf.embedding.splade_name
            )
            if not model_name:
                raise ValueError(
                    f"{sparse_model_type}_name must be provided in config"
                )

            if model_home:
                model_path = f"{model_home}/{model_name}"
            else:
                model_path = model_name

            logger.info(f"Starting up model: {model_path}")
            self.sparse_model = SPARSE_MODELS[sparse_model_type]
            self.sparse_model.startup(model_path, device=conf.env.device)

        # Initialize reranker model
        reranker_model_type = conf.reranker.model
        if reranker_model_type not in RERANKER_MODELS:
            logger.warning("Unknown reranker model, using bge-reranker-v2-m3.")
            reranker_model_type = "bge"

        model_name = (
            conf.reranker.bge_name if reranker_model_type == "bge"
            else conf.reranker.qwen3_name
        )
        if not model_name:
            raise ValueError(
                f"{reranker_model_type}_name must be provided in config"
            )

        if model_home:
            model_path = f"{model_home}/{model_name}"
        else:
            model_path = model_name

        logger.info(f"Starting up reranker model: {model_path}")
        self.reranker_model = RERANKER_MODELS[reranker_model_type]
        self.reranker_model.startup(model_path, device=conf.env.device)

    def shutdown_models(self)-> None:
        """Shutdown all loaded models."""
        if self.dense_model:
            self.dense_model.shutdown()
            self.dense_model = None

        if self.sparse_model:
            self.sparse_model.shutdown()
            self.sparse_model = None

        if self.reranker_model:
            self.reranker_model.shutdown()
            self.reranker_model = None

# Global model manager instance
model_manager = ModelManager()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifecycle."""
    # Startup
    logger.info("Starting up embedding service...")

    try:
        # Initialize models
        model_manager.startup_models()
        logger.info("Models initialized successfully")

        yield

    except Exception as e:
        logger.error(f"Failed to startup service: {e}")
        raise
    finally:
        # Shutdown
        logger.info("Shutting down embedding service...")
        model_manager.shutdown_models()
        logger.info("Service shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Embedding Service",
    description="FastAPI service for embedding and reranking operations",
    version=app_version,
    lifespan=lifespan
)

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}

@app.post("/embed")
async def embed(request: EmbeddingRequest):
    """
    Embed sentences using configured models.
    Returns binary stream containing packed embeddings.
    """
    if not model_manager.dense_model and not model_manager.sparse_model:
        raise HTTPException(
            status_code=503,
            detail="No embedding models available"
        )

    try:
        # Collect embeddings from different models
        all_embeddings = {}

        # Default to dense if nothing specified
        if not request.return_sparse and not request.return_colbert_vecs:
            request.return_dense = True

        # Orchestrate embedding requests
        dense_step = None
        sparse_step = None

        if request.return_colbert_vecs:
            if (
                model_manager.dense_type != "bge"
                and model_manager.sparse_type != "bge"
            ):
                logger.warning(
                    "ColBERT vectors requested but no BGE model available."
                )
            elif model_manager.dense_type == "bge":
                dense_step = {
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert_vecs": True,
                }
            else: # sparse_type must be "bge" here
                sparse_step = {
                    "return_dense": False,
                    "return_sparse": False,
                    "return_colbert_vecs": True,
                }

        if request.return_dense:
            if dense_step: # only when colbert vecs already requested
                dense_step["return_dense"] = True
            else:
                dense_step = {
                    "return_dense": True,
                    "return_sparse": False,
                    "return_colbert_vecs": False,
                }

        if request.return_sparse:
            if (
                model_manager.sparse_type == "bge"
                and model_manager.dense_type == "bge"
                and dense_step
            ):
                dense_step["return_sparse"] = True
            else:
                if sparse_step: # only when colbert vecs already requested
                    sparse_step["return_sparse"] = True
                else:
                    sparse_step = {
                        "return_dense": False,
                        "return_sparse": True,
                        "return_colbert_vecs": False,
                    }

        # Get embeddings
        if dense_step:
            dense_embeddings = model_manager.dense_model.encode(
                sentences=request.sentences,
                batch_size=request.batch_size or conf.embedding.batch_size,
                instruction=request.instruction,
                **dense_step,
            )
            all_embeddings.update(dense_embeddings)

        # Get sparse embeddings (if different model)
        if sparse_step:
            sparse_embeddings = model_manager.sparse_model.encode(
                sentences=request.sentences,
                batch_size=request.batch_size or conf.embedding.batch_size,
                instruction=request.instruction,
                **sparse_step,
            )
            all_embeddings.update(sparse_embeddings)

        # Unify embeddings format
        unified_embeddings = unify_embeddings(all_embeddings)
        
        # Pack to bytes
        packed_bytes = pack_unified_embeddings_to_bytes(unified_embeddings)
        
        # Return as streaming response
        return StreamingResponse(
            iter([packed_bytes]),
            media_type="application/octet-stream",
            headers={
                "Content-Type": "application/octet-stream",
                "Content-Disposition": "inline"
            }
        )

    except Exception as e:
        logger.error(f"Embedding failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Embedding failed: {str(e)}"
        )

@app.post("/rerank", response_model=RerankResponse)
async def rerank(request: RerankRequest):
    """
    Rerank documents based on relevance to query.
    Returns JSON response with scores.
    """
    if not model_manager.reranker_model:
        raise HTTPException(
            status_code=503,
            detail="No reranker model available"
        )

    try:
        # Prepare documents
        documents = request.documents
        if isinstance(documents, str):
            documents = [documents]

        # Compute relevance scores
        scores = model_manager.reranker_model.rank(
            query=request.query,
            passages=documents,
            query_instruction=request.query_instruction,
            passage_instruction=request.passage_instruction, # only for BGE
            batch_size=conf.reranker.batch_size,
            max_length=request.max_length,
            normalize=request.normalize, # only for BGE
        )

        return RerankResponse(scores=scores)

    except Exception as e:
        logger.error(f"Reranking failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Reranking failed: {str(e)}"
        )

def main(host: str | None=None, port: int | None=None, reload: bool=False):
    """Run the FastAPI app with Uvicorn."""
    import uvicorn

    # Load config for service settings
    host = host or conf.service.host
    port = port or conf.service.port

    uvicorn.run(
        "embedding_service.app:app",
        host=host,
        port=port,
        reload=reload,
        log_level="info",
    )

if __name__ == "__main__":
    main()