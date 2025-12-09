# Embedding SDK 的使用方法

`embedding-service` 可以作为一个 SDK 包被其他项目引用和使用。其中最为主要的功能即为向量化文本数据。

## 支持的 Embedding 模型

目前 `embedding-service` 支持 3 种 Embedding 模型：

- **BGE-M3**: 主力模型，同时支持密集、稀疏和 ColBERT 多向量三种方式，其中密集向量 1024 维，稀疏向量 250002 维，ColBERT 向量 1024 维；
- **Qwen3-Embedding-?B 系列**: 密集向量内嵌模型，通过配置不同的模型名称来支持三种尺寸的 Qwen3-Embedding 模型，通常建议使用 0.6B 模型即可，向量维度为 1024 维；
- **Splade-v3**: 稀疏向量内嵌模型，向量维度为 30522 维。

## 模型的使用

所有模型均采用类方法的形式进行调用，三种模型的类分别为 `BGEM3`, `Qwen3Embedding` 和 `Splade_v3`。

模型的使用步骤如下：

1. 导入模型类: `from embedding_service.model import <ModelClass>`；
2. 启动模型: 使用类方法 `<ModelClass>.startup()` 启动模型，完成加载和预热，使模型常驻内存/显存；
3. 向量化文本: 使用类方法 `<ModelClass>.encode()` 进行文本的向量化；
4. 关闭模型: 使用类方法 `<ModelClass>.shutdown()` 关闭模型，释放内存/显存，当需要更换模型或改变模型参数时需要先关闭模型。

*注意: 模型类均为类工厂单例模式，只能使用上述三个类方法，不支持实例化对象。*

### BGE-M3 模型

BGE-M3 模型支持三种向量化方式，分别为密集向量、稀疏向量和 ColBERT 多向量，是目前本项目唯一提供 ColBERT 向量化方式的模型。

**1. 启动模型**

```python
# 测试数据
queries = ["今天天气怎么样？", "人工智能的未来是什么？"]
# 引入并启动模型
from embedding_service.models import BGEM3
# 使用默认参数启动本地部署的模型，只需提供模型在本地的路径
BGEM3.startup("/path/to/local_bge_m3")
```

上述不带可选参数的调用方式，会使用模型默认参数。`BGEM3.startup()` 方法完整参数如下：

- `model_name_or_path`: 必须，模型本地路径或 Huggingface 模型名；
- `device`: 可选，默认 None，由模型自行选择设备；
- `batch_size`: 可选，默认 256；
- `normalize_embeddings`: 可选，是否对生成的向量进行归一化处理，默认 True；
- `return_dense`: 可选，是否返回密集向量，默认 True；
- `return_sparse`: 可选，是否返回稀疏向量，默认 False；
- `return_colbert`: 可选，是否返回 ColBERT 多向量，默认 False。

`model_name_or_path`, `normalize_embeddings` 和 `device` 为模型绑定参数，只能在启动时提供。其余参数均可在向量化时传入其他值。

三种向量类型是否返回的参数如果全部为 False，则会强制将 `return_dense` 设为 True。

*注意：`normalize_embeddings` 参数通常应该选择为 True，除非有特殊需求。*

**2. 向量化**

```python
# 生成嵌入向量
embeddings = BGEM3.encode(queries)
print(embeddings)
```

上述不带可选参数的调用方式，会使用模型启动时设定的参数。`BGEM3.encode()` 方法完整参数如下：

- `sentences`: 必须，需要向量化的文本列表，可以是单个字符串或一个字符串列表；
- `batch_size`: 可选，表示批处理大小；
- `return_dense`: 可选，是否返回密集向量；
- `return_sparse`: 可选，是否返回稀疏向量；
- `return_colbert`: 可选，是否返回 ColBERT 多向量。
- `instruction`: 可选，向量化指令，用于给出明确的任务类型，例如检索、分类、聚类等。

BGE-M3 是经过特定训练的“指令感知”模型，在向量化用户查询时使用合适的 `instruction` 参数有可能提升 1~5% 的召回率，但应注意：

1. 使用的指令必须经过严格测试，对同一类任务（例如检索）的用户查询提供统一的指令；
2. 检索任务推荐指令: `"为这个句子生成表示以用于检索相关文章："` ；
3. 指令最后一定要加上冒号；
4. 通常 ***禁止*** 为文档向量化提供指令。

**3. 关闭模型**

由于项目采用类单例工厂模式加载使用模型，当需要加载同一台电脑上的其他本地模型，或需要改变模型启动参数时，应当先关闭当前模型。

```python
# 关闭模型
BGEM3.shutdown()
```

模型关闭后，可以使用不同的参数再次启动。

### Qwen3-Embedding 系列模型

Qwen3-Embedding 系列模型共有 0.6B、4B、8B 三个尺寸版本，生成向量维度分别为 1024、2560 和 4096 维，精度高于 BGE-M3 模型，但只支持密集向量。

Qwen3-Embedding 系列模型使用类 `Qwen3Embedding` 启动、使用和关闭，用法与 `BGEM3` 类一致，`startup()` 和 `encode()` 的可选参数较少。

- `Qwen3Embedding.startup()` 无 `return_dense`, `return_sparse`, `return_colbert_vecs` 参数，模型仅支持密集向量。
- `Qwen3Embedding.startup()` 无 `normalize_embeddings` 参数，模型强制归一化所有向量。
- `Qwen3Embedding.startup()` 无 `batch_size` 参数，该参数只能在 `encode()` 时提供。
- `Qwen3Embedding.encode()` 无 `return_dense`, `return_sparse`, `return_colbert_vecs` 参数。

Qwen3-Embedding 系列同样是经过特定训练的“指令感知”模型，同样可以在 `encode()` 方法中提供 `instruction` 指令以提升性能，用法和 BGE-M3 模型相同，也同样需要注意：

1. 使用的指令必须经过严格测试，对同一类任务（例如检索）的用户查询提供统一的指令；
2. 检索任务推荐指令: `"为这个句子生成表示以用于检索相关文章："` ；
3. 指令最后一定要加上冒号；
4. 通常 ***禁止*** 为文档向量化提供指令。

### Splade-v3 模型

Splade-v3 是一个稀疏向量内嵌模型，只能生成稀疏向量。

Splade-v3 模型使用类 `Splade_v3` 启动和使用，与 `Qwen3Embedding` 的用法完全相同。

## 返回值

TODO: continue...
