#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-03 08:20
# Filename         : __init__.py
# Project          : HuRAG/embedding-service
#*********************************************************************

__version__ = "0.1.0"
__author__ = "Libin, QuestYard HuRAG Team"
__description__ = "SDK and API for embedding and reranker models"

import yaml
from pathlib import Path

# -- Global Variables --

conf = None     # Global configurations
logger = None   # Global logger

# -- Initialization --

with open(Path.cwd() / "embedding-service.yaml", "r", encoding="utf-8") as f:
    conf = yaml.safe_load(f)

from .utilities import dict_to_namespace
conf = dict_to_namespace(conf or {})
conf.env.device = conf.env.device or "cpu"
conf.embedding.mode = conf.embedding.mode or "bge"
conf.embedding.batch_size = conf.embedding.batch_size or 16
conf.reranker.mode = conf.reranker.mode or "bge"
conf.reranker.batch_size = conf.reranker.batch_size or 4

import logging
import logging.handlers

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

__all__ = [
    "conf",
    "logger",
]

