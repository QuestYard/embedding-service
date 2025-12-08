# Embedding SDK 的使用方法

`embedding-service` 可以作为一个 SDK 包被其他项目引用和使用。其中最为主要的功能即为向量化文本数据。

## 支持的 Embedding 模型

目前 `embedding-service` 支持 3 种 Embedding 模型：

- **BGE-M3**: 主力模型，同时支持密集、稀疏和 ColBERT 多向量三种方式，其中密集向量 1024 维，稀疏向量 250002 维，ColBERT 向量 1024 维；
- **Qwen3-Embedding 系列**: 密集向量内嵌模型，通过配置不同的模型名称来支持三种尺寸的 Qwen3-Embedding 模型，通常建议使用 0.6B 模型即可，向量维度为 1024 维；
- **Splade-v3**: 稀疏向量内嵌模型，向量维度为 30522 维。

## 模型的使用

所有模型均采用类方法的形式进行调用，三种模型的类分别为 `BGEM3`, `Qwen3Embedding` 和 `Splade_v3`。

模型的使用步骤如下：

1. 导入模型类: `from embedding_service.model import <ModelClass>`；
2. 启动模型: 使用类方法 `<ModelClass>.startup()` 启动模型，完成加载和预热，使模型常驻内存/显存；
3. 向量化文本: 使用类方法 `<ModelClass>.encode()` 进行文本的向量化；
4. 关闭模型: 使用类方法 `<ModelClass>.shutdown()` 关闭模型，释放内存/显存，当需要更换模型或改变模型参数时需要先关闭模型。

*注意事项*: 模型类均为类工厂单例模式，只能使用上述三个类方法，不支持实例化对象。

### BGE-M3 模型使用示例

BGE-M3 模型支持三种向量化方式，分别为密集向量、稀疏向量和 ColBERT 多向量，是目前本项目唯一提供 ColBERT 向量化方式的模型。

在使用前准备好配置文件和测试数据，配置文件名为 `embedding-service.yaml`，需要放置在当前工作目录下：

[embedding-service.yaml 示例配置文件](./embedding-service.yaml)。

```python
# 测试数据
queries = ["今天天气怎么样？", "人工智能的未来是什么？"]
# 引入并启动模型
from embedding_service.models import BGEM3
# 使用类方法启动模型
BGEM3.startup()
```

上述不带参数的启动方式，会使用默认参数加载模型，其中模型名称和路径从配置文件中读取，其他参数取默认值。

`BGEM3.startup()` 方法完整参数如下：

- `model_name_or_path`: 默认使用 `conf.env.model_home + "/" + conf.embedding.bge_name` 配置的模型路径名，可自行传入其他路径名；
- `device`: 默认使用 `conf.env.device` 配置的设备，可自行传入其他设备，如 "cpu", "cuda:0" 等；
- `batch_size`: 默认使用 `conf.embedding.batch_size` 配置的批处理大小，可自行传入其他大小；
- `normalize_embeddings`: `BGEM3.startup()` 特有参数，表示是否对生成的向量进行归一化处理，默认为 True；
- `return_dense`: BGEM3 特有参数，表示是否返回密集向量，默认为 True；
- `return_sparse`: BGEM3 特有参数，表示是否返回稀疏向量，默认为 False；
- `return_colbert`: BGEM3 特有参数，表示是否返回 ColBERT 多向量，默认为 False。

`model_name_or_path`, `normalize_embeddings` 和 `device` 为模型绑定参数，只能在启动时提供，均参数可在调用向量化方法 `BGEM3.encode()` 时传入其他值。

三种向量类型是否返回的参数如果均设为 False，则会强制将 `return_dense` 设为 True。

*注意：`normalize_embeddings` 参数通常应该选择为 True，除非有特殊需求。*

```python
# 生成嵌入向量
embeddings = BGEM3.encode(queries)
print(embeddings)
```

上述不带可选参数的调用方式，会使用启动时绑定的参数进行向量化。

`BGEM3.encode()` 方法完整参数如下：

- `sentences`: 需要向量化的文本列表，可以是单个字符串或一个字符串列表；
- `batch_size`: 可选参数，表示批处理大小，不提供或 None 表示使用启动时绑定的批处理大小；
- `return_dense`: 可选参数，是否返回密集向量，不提供或 None 表示使用启动时绑定的参数；
- `return_sparse`: 可选参数，是否返回稀疏向量，不提供或 None 表示使用启动时绑定的参数；
- `return_colbert`: 可选参数，是否返回 ColBERT 多向量，不提供或 None 表示使用启动时绑定的参数。

*注意：`batch_size` 参数在纯 CPU 环境或显存小于 6GB 的 GPU 环境下不应过大。*