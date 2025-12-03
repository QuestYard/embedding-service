#*********************************************************************
# Author           : Libin
# Company          : huz.zj.yc
# Last modified    : 2025-12-03 10:57
# Filename         : bge_m3.py
# Project          : HuRAG/embedding-service
#*********************************************************************
from .. import conf
from .abstract_models import AbstractEmbedder, AbstractReranker

# from FlagEmbedding import BGEM3FlagModel

class BGEM3:
    MODEL = None
    RETURN_DENSE: bool
    RETURN_SPARSE: bool
    RETURN_COLBERT_VECS: bool

    @classmethod
    def encode(
        cls,
        sentences: str|list[str],
        normalize_embeddings: bool|None,
        return_dense: bool|None,
        return_sparse: bool|None,
        return_colbert_vecs: bool|None,
    ):
        if not sentences:
            return {
                "dense_vecs": None,
                "lexical_weights": None,
                "colbert_vecs":None
            }

        if cls.MODEL is None:
            cls.startup()

    return {}

    @classmethod
    def startup(
        cls,
        model_name_or_path: str|None=None,
        device: str|None=None,
        batch_size: int|None=None,
        normalize_embeddings: bool|None=None,
        return_dense: bool|None=None,
        return_sparse: bool|None=None,
        return_colbert_vecs: bool|None=None,
    ):
        cls.MODEL_NAME_OR_PATH = (
            model_name_or_path if model_name_or_path else
            conf.env.model_home + "/" + conf.embedding.bge_name
        )
        cls.DEVICE = device or conf.env.device
        cls.BATCH_SIZE = batch_size or conf.embedding.batch_size
        cls.NORMALIZE_EMBEDDINGS = (
            normalize_embeddings
            if normalize_embeddings is not None
            else True
        )
        cls.RETURN_DENSE = return_dense if return_dense is not None else True
        cls.RETURN_SPARSE = (
            return_sparse
            if return_sparse is not None
            else True
        )
        cls.RETURN_COLBERT_VECS = (
            return_colbert_vecs
            if return_colbert_vecs is not None
            else False
        )
        if (
            not cls.RETURN_DENSE and
            not cls.RETURN_SPARSE and
            not cls.RETURN_COLBERT_VECS
        ):
            cls.RETURN_DENSE = True

        cls.MODEL = cls.MODEL_NAME_OR_PATH # HACK

        print(f"model '{cls.MODEL_NAME_OR_PATH}' startup")

def ef(
    sentences: str|list[str],
    normalize_embeddings: bool|None=None,
    return_dense: bool|None=None,
    return_sparse: bool|None=None,
    return_colbert_vecs: bool|None=None,
):
    BGEM3.encode(
        sentences,
        normalize_embeddings,
        return_dense,
        return_sparse,
        return_colbert_vecs,
    )

def startup():
    pass

#     @classmethod
#     def invoke(
#         cls,
#         sentences: str|list[str],
#         batch_size=16,
#         device=None,
#         return_dense=True,
#         return_sparse=True,
#         return_colbert_vecs=False,
#     ):
#         if cls.model is None:
#             cls.model = BGEM3EmbeddingFunction(
#                 model_name=os.environ["MODEL_HOME"] + "/BAAI/bge-m3",
#                 use_fp16=False,
#                 device=device or os.environ["LOCAL_DEVICE"],
#                 batch_size=batch_size,
#                 return_dense=return_dense,
#                 return_sparse=return_sparse,
#                 return_colbert_vecs=return_colbert_vecs,
#             )
#         return cls.model(texts if isinstance(texts, list) else [texts])

# def bge_m3_ef(
#     texts,
#     device=None,
#     batch_size=16,
#     return_dense=True,
#     return_sparse=True,
#     return_colbert_vecs=False,
# ):
#     return BGEM3_MILVUS.invoke(
#         texts,
#         device,
#         batch_size,
#         return_dense,
#         return_sparse,
#         return_colbert_vecs,
#     )


