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
```

**torch源配置**

仓库源代码 `pyproject.toml` 中是根据项目开发环境配置的 `torch` 源。将仓库克隆到本地后，首先需要根据本地服务器的软硬件配置修改。

```bash
cd <src-dir>
# 修改 pyproject.toml 中 torch 的 sources
vim pyproject.toml
```

需要修改的配置项如下：

```toml
[tool.uv.sources]
torch = [
    { index = "pytorch-cpu", marker = "sys_platform == 'linux'" },
    { index = "pytorch-cu130", marker = "sys_platform == 'win32'" },
]
```

根据本地服务器的硬件配置，通过 `marker` 精确定义本地软硬件环境，从而指定需要使用的 `pytorch` 类型。

严格遵循 PEP 和 uv 配置的规范编写 `marker` 约束，规范依据：

- [Using uv with PyTorch](https://docs.astral.sh/uv/guides/integration/pytorch/)
- [PEP 508 – Dependency specification for Python Software Packages](https://peps.python.org/pep-0508/)
- [Dependency specifiers](https://packaging.python.org/en/latest/specifications/dependency-specifiers/)

**同步环境**

配置修改完成后，同步 uv 环境即可完成本地安装。

```bash
# 若不使用 --no-dev 参数，将额外安装 jupyter 和 matplotlib 两个开发依赖
uv sync --no-dev
```

### As an SDK package

`embedding-service` 也可以作为一个 SDK 包被其他项目引用和使用。

建议采用 uv 进行项目管理，运行命令 `uv add -e /path/to/embedding-service` 进行可编辑安装。采用 pip 管理的项目，可以通过 `pip install -e /path/to/embedding-service` 命令进行安装。目前尚不支持 PyPI 发布。

*作为一个 SDK 包使用时，无需提供 `embedding-service.yaml` 配置文件。若提供配置文件，其中的配置信息不起作用，需要用户项目自行管理模型参数。*

SDK 说明详见：

- [Embedding SDK 的使用方法](./docs/sdk_usages/1_embedding.md)
- [Reranker SDK 的使用方法](./docs/sdk_usages/2_reranker.md)

### As a Micro-Service

TODO *constructing...*

#### Instance Setup

**创建服务实例目录**

```bash
mkdir <working-dir>
cd <working-dir>
cp <src-dir>/embedding-service.yaml.sample embedding-service.yaml
cp <src-dir>/startup.sh .
```

**配置服务**

作为微服务时，必须在项目工作目录下提供配置文件 `embedding-service.yaml` 如下：

```yaml
env:            # SERVER ENVIRONMENT
  device:       # cpu, cuda:x, or None (automatically choose by model)
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

TODO *constructing...*
