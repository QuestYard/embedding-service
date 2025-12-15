# Embedding Service API 文档

本文档详细说明了 Embedding Service 提供的 API 接口以及 Python 客户端的使用方法。

## 1. API 接口详解

服务提供两个主要接口：文本向量化 (`/embed`) 和 重排序 (`/rerank`)。

### 1.1 文本向量化 (Embed)

将输入的文本转换为向量表示。支持稠密向量 (Dense)、稀疏向量 (Sparse/Splade) 和 ColBERT 向量。

- **URL**: `/embed`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (Request Body)

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `sentences` | `List[str]` or `str` | 是 | - | 需要向量化的句子或句子列表 |
| `batch_size` | `int` | 否 | `None` | 推理时的批次大小 |
| `return_dense` | `bool` | 否 | `True` | 是否返回稠密向量 |
| `return_sparse` | `bool` | 否 | `False` | 是否返回稀疏向量 |
| `return_colbert_vecs` | `bool` | 否 | `False` | 是否返回 ColBERT 向量 |
| `instruction` | `str` | 否 | `None` | 针对查询的指令（通常用于非对称搜索的 Query 端） |

#### 响应 (Response)

- **Content-Type**: `application/octet-stream`
- **说明**: 返回值为二进制流，包含打包后的向量数据 (numpy arrays) 和元数据。由于格式较为复杂，**强烈建议使用提供的 SDK (`AsyncEmbeddingClient`) 进行调用和解析**。

#### httpx 直接调用示例

如果你必须直接使用 HTTP 客户端调用，请注意响应体是二进制数据。

```python
import httpx
import asyncio

async def call_embed_raw():
    url = "http://localhost:8000/embed"
    payload = {
        "sentences": ["Hello world", "AI is great"],
        "return_dense": True,
        "return_sparse": False
    }
    async with httpx.AsyncClient() as client:
        # 注意：这里使用了 stream 模式，因为响应可能较大
        async with client.stream("POST", url, json=payload) as response:
            if response.status_code == 200:
                # 读取所有二进制数据
                data = await response.aread()
                print(f"Received {len(data)} bytes of binary data")
                # 注意：这里得到的 data 需要使用 adapters.unpack_unified_embeddings_from_bytes 进行解包
            else:
                print(f"Error: {response.status_code}, {response.text}")

if __name__ == "__main__":
    asyncio.run(call_embed_raw())
```

---

### 1.2 重排序 (Rerank)

计算查询 (Query) 与一组文档 (Documents) 之间的相关性分数。

- **URL**: `/rerank`
- **Method**: `POST`
- **Content-Type**: `application/json`

#### 请求参数 (Request Body)

| 参数名 | 类型 | 必填 | 默认值 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| `query` | `str` | 是 | - | 查询语句 |
| `documents` | `List[str]` or `str` | 是 | - | 待排序的文档列表 |
| `query_instruction` | `str` | 否 | `None` | 查询指令 |
| `passage_instruction` | `str` | 否 | `None` | 文档指令 |
| `batch_size` | `int` | 否 | `None` | 推理时的批次大小 |
| `max_length` | `int` | 否 | `None` | 上下文最大长度 |
| `normalize` | `bool` | 否 | `None` | 是否对分数进行归一化 (Sigmoid) |

#### 响应 (Response)

- **Content-Type**: `application/json`

返回 JSON 对象：
```json
{
  "scores": [0.98, 0.45, 0.12]
}
```

#### httpx 直接调用示例

```python
import httpx
import asyncio

async def call_rerank_raw():
    url = "http://localhost:8000/rerank"
    payload = {
        "query": "What is AI?",
        "documents": [
            "AI stands for Artificial Intelligence",
            "The weather is nice today"
        ],
        "normalize": True
    }
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=payload)
        if response.status_code == 200:
            result = response.json()
            print("Scores:", result["scores"])
        else:
            print(f"Error: {response.text}")

if __name__ == "__main__":
    asyncio.run(call_rerank_raw())
```

---

## 2. Python 客户端 (AsyncEmbeddingClient)

为了简化调用，特别是为了自动处理 `/embed` 接口的二进制解包，项目提供了 `AsyncEmbeddingClient`。

### 2.1 引入与初始化

客户端设计为异步上下文管理器 (`async with`)。

```python
from embedding_service.async_embedding_client import AsyncEmbeddingClient

# 初始化参数
# base_url: 服务地址
# timeout: 超时时间 (默认 300.0 秒)
client = AsyncEmbeddingClient(base_url="http://localhost:8000")
```

### 2.2 方法说明

#### `embed` 方法

```python
async def embed(
    self,
    sentences: str | list[str],
    batch_size: int | None = None,
    return_dense: bool = True,
    return_sparse: bool = False,
    return_colbert_vecs: bool = False,
    instruction: str | None = None
) -> tuple[dict, EmbeddingPayloadMeta]
```

**返回值**:
返回一个元组 `(embeddings, meta)`:
- `embeddings (dict)`: 包含向量数据的字典。键可能包含 `dense_vecs`, `sparse_vecs`, `colbert_vecs` 等，值为 numpy 数组或 CSR 矩阵。
- `meta (EmbeddingPayloadMeta)`: 包含数据形状、类型等元信息的对象。

#### `rerank` 方法

```python
async def rerank(
    self,
    query: str,
    documents: str | list[str],
    query_instruction: str | None = None,
    passage_instruction: str | None = None,
    batch_size: int | None = None,
    max_length: int | None = None,
    normalize: bool | None = None
) -> RerankResponse
```

**返回值**:
- `RerankResponse`: 包含 `scores` (List[float]) 属性的对象。

### 2.3 完整使用示例

```python
import asyncio
from embedding_service.async_embedding_client import AsyncEmbeddingClient

async def main():
    base_url = "http://localhost:8000"
    
    async with AsyncEmbeddingClient(base_url) as client:
        # 1. 测试 Embed 接口
        print("--- Testing Embed ---")
        sentences = ["Hello world", "Machine learning is amazing"]
        
        embeddings, meta = await client.embed(
            sentences=sentences,
            return_dense=True,
            return_sparse=False
        )
        
        print(f"Meta info: {meta}")
        
        if embeddings.get("dense_vecs") is not None:
            dense_vecs = embeddings["dense_vecs"]
            print(f"Dense shape: {dense_vecs.shape}")
            # 打印第一个句子的前5维
            print(f"Vector sample: {dense_vecs[0][:5]}")

        # 2. 测试 Rerank 接口
        print("\n--- Testing Rerank ---")
        query = "What is AI?"
        docs = [
            "Artificial intelligence is the simulation of human intelligence.",
            "Python is a programming language."
        ]
        
        result = await client.rerank(
            query=query,
            documents=docs,
            normalize=True
        )
        
        print(f"Query: {query}")
        for doc, score in zip(docs, result.scores):
            print(f"Score: {score:.4f} | Doc: {doc}")

if __name__ == "__main__":
    asyncio.run(main())
```
