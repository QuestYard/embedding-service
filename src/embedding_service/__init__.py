__version__ = "0.2.1"
__author__ = "Libin, QuestYard HuRAG Team"
__description__ = "SDK and API for embedding and reranker models"

import yaml
import logging
from pathlib import Path
from dotenv import load_dotenv
from typing import Any

load_dotenv(Path.cwd() / ".env")

# -- Global Variables --

logger: logging.Logger
conf: Any

# -- Initialization --

logger = logging.getLogger("hurag-embedding-svr")
logger.propagate = False
logger.setLevel(logging.DEBUG)
fmt = logging.Formatter("%(asctime)s [%(name)s] %(levelname)s - %(message)s")
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

from .utilities import dict_to_namespace

try:
    with open(Path.cwd() / "embedding-service.yaml", "r", encoding="utf-8") as f:
        conf = yaml.safe_load(f)
    conf = dict_to_namespace(conf)
    conf.env.model_home = conf.env.model_home or ""
    conf.embedding.dense_model = conf.embedding.dense_model or "bge"
    conf.embedding.sparse_model = conf.embedding.sparse_model or "bge"
    conf.embedding.batch_size = conf.embedding.batch_size or 16
    conf.reranker.model = conf.reranker.model or "bge"
    conf.reranker.batch_size = conf.reranker.batch_size or 4
    conf.service.host = conf.service.host or "0.0.0.0"
    conf.service.port = conf.service.port or 8765

    conf.env.model_home = Path(conf.env.model_home).expanduser().resolve().as_posix()
    if conf.reranker.model.lower() == "glm":
        load_dotenv(Path.cwd() / ".env")
except:
    conf = None
    raise

from .async_embedding_client import AsyncEmbeddingClient

__all__ = [
    "conf",
    "logger",
    "AsyncEmbeddingClient",
]
