#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-06 23:37
# Filename         : __init__.py
# Project          : HuRAG/embedding-service
#*********************************************************************
from .bge_m3 import ef as bge_m3_ef, shutdown as bge_m3_shutdown
from .qwen3_embedding import ef as qwen3_ef, shutdown as qwen3_shutdown

__all__ = [
    "bge_m3_ef",
    "bge_m3_shutdown",
    "qwen3_ef",
    "qwen3_shutdown",
]
