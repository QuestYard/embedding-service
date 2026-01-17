__version__ = "0.1.1"
__author__ = "Libin, QuestYard HuRAG Team"
__description__ = "SDK and API for embedding and reranker models"

import yaml
from pathlib import Path

# -- Global Variables --

conf = None     # Global configurations
logger = None   # Global logger

# -- Initialization --

from .utilities import dict_to_namespace
try:
    with open(Path.cwd()/"embedding-service.yaml", "r", encoding="utf-8") as f:
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
except:
    pass

if conf.reranker.model.lower() == "glm":
    from dotenv import load_dotenv
    load_dotenv()

import logging

logger = logging.getLogger("hurag-embedding-svr")
logger.propagate = False
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "%(asctime)s [%(name)s] %(levelname)s - %(message)s"
)
console_handler = logging.StreamHandler()
console_handler.setFormatter(fmt)
console_handler.setLevel(logging.INFO)
logger.addHandler(console_handler)

from .async_embedding_client import AsyncEmbeddingClient

__all__ = [
    "conf",
    "logger",
    "AsyncEmbeddingClient",
]

