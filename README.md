# Introduction

`embedding-service` 是一个基于 FastAPI 构建的微服务，用于向 RAG 应用提供文本向量内嵌（embedding）和重排序（rerank）服务。通过在服务器上加载、预热并保持本地化模型，向 RAG 应用提供高效的内嵌和重排服务，避免重复加载带来的耗时。

## Key Features

### Embedding

支持 BGE-M3 模型、Qwen3-Embedding 系列模型和 Splade-V3 模型的向量内嵌，支持可直接用于 Milvus 混合搜索的基于 BGE-M3 模型的双向量内嵌。

**支持的模型**

|Model             |Dense Embedding|Sparse Embedding (Lexical Weight)|Multi-Vector (ColBERT)|
|------------------|---------------|---------------------------------|----------------------|
|BGE-M3            |       ✅      |               ✅                |          ✅          |
|Qwen3-Embedding-?B|       ✅      |               ❌                |          ❌          |
|Splade-V3         |       ❌      |               ✅                |          ❌          |

其中 Qwen3-Embedding 模型的尺寸包括 0.6B, 4B, 8B 三种，根据实际下载使用的模型尺寸，在配置文件中配置，*每个服务实例只能选择其中一种尺寸*。

**模型选择**

在服务配置文件中选择要使用的 embedding 模型：

- `dense_model`: `bge` 或者 `qwen3`，默认使用 BGE-M3 模型，此时可支持 ColBERT 多向量内嵌，使用 Qwen3 模型时不支持 ColBERT；
- `sparse_model`: `bge` 或者 `splade`；

目前只提供 BGE-M3 一种 ColBERT 多向量内嵌模型，无需在配置文件中配置。 `dense_model` 或 `sparse_model` 任一设置为 `bge` 时可提供 ColBERT 多向量内嵌，否则即不支持。

**调用参数**

- 通过参数 `return_dense`, `return_sparse`, `return_colbert` 来指定内嵌的类型，对于当前模式不支持的类型，参数不起作用，返回 None。

### Reranker

支持 BGE-Reranker-V2-M3 模型和 Qwen3-Reranker 系列模型，默认使用 BGE-Reranker-V2-M3 模型。在配置文件中设置 `reranker.model` 为 `bge` 或 `qwen3` 进行选择。

*重排模型与内嵌模型不必要相互对应*。

## Quick Start

### Installation

项目使用 uv 构建，使用源码方式安装，需要具备以下环境：

- uv
- git
- python >= 3.12
- 本地模型下载，支持 HuggingFace 和 ModelScope 下载的模型

**从源码安装**

```bash
git clone https://github.com/QuestYard/embedding-service.git <src-dir>
cd <src-dir>
uv sync --no-dev
```

### Instance Setup

**创建服务实例目录**

```bash
mkdir <working-dir>
cd <working-dir>
cp <src-dir>/embedding-service.yaml.sample embedding-service.yaml
cp <src-dir>/startup.sh .
```

**配置服务**

修改工作目录下的配置文件 `embedding-service.yaml` ：

```yaml
env:            # SERVER ENVIRONMENT
  device:       # cpu (default) or cuda
  model_home:   # /path/to/model_home, e.g. /home/user/.cache/modelscope/hub/models
embedding:      # EMBEDDING MODEL CONFIGURATIONS
  dense_model:  # bge (default) or qwen3
  sparse_model: # bge (default) or splade
  bge_name:     # name (path name) of the local bge-m3 model
  qwen3_name:   # name (path name) of the local qwen3 embedding model
  splade_name:  # name (path name) of the local splade-v3 model
  batch_size:   # >=4 (default 16)
reranker:       # RERANKER MODEL CONFIGURATIONS
  model:        # bge (default) or qwen3
  bge_name:     # name (path name) of the local bge-reranker-v2-m3 model
  qwen3_name:   # name (directory name) of the local qwen3 reranker model
  batch_size:   # >=4 (default 4)
service:        # MICRO-SERVICE CONFIGURATIONS
  host:         # 0.0.0.0 (default), ip-addr for allowed hosts 
  port:         # 8765 (default), port of service
```

**启动与测试**

配置完成后，进入工作目录执行启动脚本即可启动服务。

```bash
cd <working-dir>
./startup.sh
```

