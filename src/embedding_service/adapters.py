from __future__ import annotations
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
    from torch import Tensor
    from .schemas import EmbeddingPayloadMeta


# --- Adapters for Milvus vector database ---

def sparse_tensor_to_csr_matrix(t: Tensor)-> csr_matrix:
    """
    Convert a sparse torch.Tensor to scipy.sparse.csr_matrix.
    
    It's used to convert sparse embeddings from models like Splade-v3 into
    a format suitable for insertion into Milvus and compute similarity.

    Args:
        t (Tensor): A sparse torch.Tensor, e.g., returned by Splade-v3 model.

    Returns:
        csr_matrix: The corresponding scipy.sparse.csr_matrix.
    """
    from scipy.sparse import csr_matrix
    import numpy as np

    if t is None or t.numel() == 0:
        return csr_matrix((1, 0))  # empty sparse matrix

    coo = t.to_sparse().coalesce()
    values = coo.values().cpu().numpy()
    indices = coo.indices().cpu().numpy()
    shape = tuple(coo.size())
    return csr_matrix(
        (values, (indices[0], indices[1])),
        shape=shape,
        dtype=np.float32
    )

def lexical_weights_to_csr_matrices(
    lw: list[dict[str, float]],
    vocab_size: int = 250002,     # dim of BGE-M3 model's sparse vector
)-> csr_matrix:
    """
    Convert lexical weights dictionary list to scipy.sparse.csr_matrix.

    It's used to convert lexical weights from models like BGEM3 into a format
    suitable for insertion into Milvus and compute similarity.

    Args:
        lw (list[dict[str, float]]): A list of lexical weights dictionaries,
            e.g., returned by BGEM3 model.
        dim (int): Dimension of the sparse vector.
            Default is 250002 for BGE-M3. It can also be aquired by
            `BGEM3._model.model.vocab_size` or `len(BGEM3._model.tokenizer)`.
            vocab_size = 30522 for Splade-V3 model.

    Returns:
        csr_matrix: The corresponding scipy.sparse.csr_matrix.
    """
    from scipy.sparse import csr_matrix
    import numpy as np

    n_rows = len(lw)
    if n_rows == 0:
        return csr_matrix((0, vocab_size), dtype=np.float32)

    total_nnz = 0
    for d in lw:
        total_nnz += len(d)

    data = np.empty(total_nnz, dtype=np.float32)
    indices = np.empty(total_nnz, dtype=np.int32)
    indptr = np.empty(n_rows + 1, dtype=np.int32)

    p = 0
    for i, d in enumerate(lw):
        indptr[i] = p
        if d:
            for k, v in d.items():
                indices[p] = int(k)
                data[p] = v
                p += 1

    indptr[n_rows] = p

    if p != total_nnz:
        data = data[:p]
        indices = indices[:p]

    return csr_matrix((data, indices, indptr), shape=(n_rows, vocab_size))

def unify_embeddings(embeddings: dict)-> dict:
    """
    Unify embeddings dictionary from different models into a standard format.

    The returned dictionary will have the following keys:
    - "dense_vecs": numpy.ndarray | None
    - "sparse_vecs": scipy.sparse.csr_matrix | None
    - "colbert_vecs": list[numpy.ndarray] | None

    It's used to prepare http responses in a consistent format regardless of
    the underlying model used for encoding.

    Args:
        embeddings (dict): The embeddings dictionary returned by a model's
            encode method.

    Returns:
        dict: A unified embeddings dictionary.
    """
    results = {}
    results["dense_vecs"] = embeddings.get("dense_vecs", None)
    results["sparse_vecs"] = None
    if "lexical_weights" in embeddings:
        lw = embeddings["lexical_weights"]
        if isinstance(lw, list):
            results["sparse_vecs"] = lexical_weights_to_csr_matrices(lw)
        else:
            results["sparse_vecs"] = sparse_tensor_to_csr_matrix(lw)
    results["colbert_vecs"] = embeddings.get("colbert_vecs", None)

    return results

def pack_unified_embeddings_to_bytes(
    unified_embeddings: dict,
)-> bytes:
    """
    Pack three embedding parts into a compressed .npz bytes stream.

    The input dictionary can be obtained via `unify_embeddings` function,
    it should have the following keys:
    - "dense_vecs": numpy.ndarray | None
    - "sparse_vecs": scipy.sparse.csr_matrix | None
    - "colbert_vecs": list[numpy.ndarray] | None

    The returned dictionary will only include non-None embeddings.

    Args:
        unified_embeddings (dict): The unified embeddings dictionary.

    Returns:
        bytes: A .npz packed bytes object containing the embeddings.
    """
    import io
    import json
    import numpy as np

    dense = unified_embeddings.get("dense_vecs", None)
    sparse = unified_embeddings.get("sparse_vecs", None)
    colbert = unified_embeddings.get("colbert_vecs", None)
    meta = {
        "has_dense": bool(dense is not None),
        "has_sparse": bool(sparse is not None),
        "has_colbert": bool(colbert is not None),
        "format_version": "npz_v1",
    }

    # prepare arrays to save in np.savez_compressed
    save_dict = {}

    if dense is not None:
        arr = np.asarray(dense, dtype=np.float32)
        save_dict["dense_data"] = arr.ravel()

        meta.update(
            {"dense_shape": tuple(arr.shape), "dense_dtype": str(arr.dtype)}
        )

    if sparse is not None:
        save_dict["sparse_data"] = np.asarray(sparse.data, dtype=np.float32)
        save_dict["sparse_indices"] = np.asarray(sparse.indices, dtype=np.int32)
        save_dict["sparse_indptr"] = np.asarray(sparse.indptr, dtype=np.int32)

        meta.update(
            {
                "sparse_meta": {
                    "nnz": int(sparse.nnz),
                    "shape": tuple(sparse.shape),
                    "dtype": str(sparse.data.dtype),
                }
            }
        )

    if colbert is not None:
        meta_col_shapes = []
        for i, v in enumerate(colbert):
            arr = np.asarray(v)
            save_dict[f"colbert_{i}"] = arr.ravel()
            meta_col_shapes.append(tuple(arr.shape))
        meta.update(
            {
                "colbert_meta": {
                    "count": len(colbert),
                    "shapes": meta_col_shapes,
                    "dtype": str(colbert[0].dtype),
                }
            }
        )

    # save meta as JSON bytes
    meta_json = json.dumps(meta, separators=(",", ":"), ensure_ascii=False)
    save_dict["meta"] = np.array(meta_json.encode("utf-8"), dtype="S")

    # write compressed npz to buffer
    buf = io.BytesIO()
    np.savez_compressed(buf, **save_dict)
    return buf.getvalue()

def unpack_unified_embeddings_from_bytes(
    npz_bytes: bytes
)-> tuple[dict, EmbeddingPayloadMeta]:
    """
    Unpack unified embeddings and meta from a compressed .npz bytes stream.

    The returned dictionary will have the following keys:
    - "dense_vecs": numpy.ndarray | None
    - "sparse_vecs": scipy.sparse.csr_matrix | None
    - "colbert_vecs": list[numpy.ndarray] | None

    The returned meta is an instance of EmbeddingPayloadMeta schema.

    It's used to parse http request payloads containing packed embeddings.

    Args:
        npz_bytes (bytes): A .npz packed bytes object containing the embeddings.

    Returns:
        tuple: A tuple containing:
            - dict: A unified embeddings dictionary.
            - EmbeddingPayloadMeta: The metadata of the embeddings.
    """
    import io
    import numpy as np
    from scipy.sparse import csr_matrix
    from .schemas import EmbeddingPayloadMeta

    buf = io.BytesIO(npz_bytes)
    npz = np.load(buf, allow_pickle=False)

    meta_json = npz["meta"].tolist().decode("utf-8")
    meta = EmbeddingPayloadMeta.model_validate_json(meta_json)

    dense = None
    sparse = None
    colbert = None

    if meta.has_dense:
        dt = meta.dense_dtype
        shape = tuple(meta.dense_shape)
        raw = npz["dense_data"]
        dense = np.asarray(raw, dtype=dt).reshape(shape)

    if meta.has_sparse:
        data = npz["sparse_data"].astype(meta.sparse_meta.dtype)
        indices = npz["sparse_indices"].astype(np.int32)
        indptr = npz["sparse_indptr"].astype(np.int32)
        shape = tuple(meta.sparse_meta.shape)
        sparse = csr_matrix((data, indices, indptr), shape=shape)

    if meta.has_colbert:
        cm = meta.colbert_meta
        count = int(cm.count)
        dtype = cm.dtype
        colbert = []
        for i in range(count):
            shape = tuple(cm.shapes[i])
            raw = npz[f"colbert_{i}"]
            arr = np.asarray(raw, dtype=dtype).reshape(shape)
            colbert.append(arr)

    return {
        "dense_vecs": dense,
        "sparse_vecs": sparse,
        "colbert_vecs": colbert,
    }, meta

