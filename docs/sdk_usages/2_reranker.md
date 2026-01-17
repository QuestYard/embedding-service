# Reranker SDK 的使用方法

`embedding-service` 作为 SDK 包被其他项目引用和使用时，另一主要功能为“查询-段落”对的重排序。

## 支持的 Reranker 模型

目前 `embedding-service` 支持 3 种 Reranker 模型：

- **BGE-Reranker-v2-m3**: 主力模型；
- **Qwen3-Reranker-?B 系列**: 通常建议使用 0.6B 模型，重排速度较快。
- **GLM-Rerank**: 智谱AI推出的收费重排序模型，需要在 `.env` 中配置 GLM Token。

## 模型的使用

所有模型均采用类方法的形式进行调用，两种模型对应的类名分别为 `BGEReranker` 和 `Qwen3Reranker`。

模型的使用步骤如下：

1. 导入模型类: `from embedding_service.models import <ModelClass>`；
2. 启动模型: 使用类方法 `<ModelClass>.startup()` 启动模型，完成加载和预热，使模型常驻内存/显存；
3. 重排序: 使用类方法 `<ModelClass>.rank()` 进行“查询-段落”对的重排序，返回各段落对应的得分；
4. 关闭模型: 使用类方法 `<ModelClass>.shutdown()` 关闭模型，释放内存/显存，当需要更换模型或改变模型参数时需要先关闭模型。

*注意: 模型类均为类工厂单例模式，只能使用上述三个类方法，不支持实例化对象。*

### BGE-Reranker-v2-m3 模型

使用 BGE-M3 模型进行向量化时，重排序阶段建议使用 BGE-Reranker-v2-m3 模型。

**1. 启动模型**

```python
# 测试数据
query = "今天天气怎么样？"
passages = [
    "今天天气很好，适合出去游玩。",
    "人工智能的发展日新月异，前景广阔。",
    "今天可能会下雨，记得带伞。"
]

# 引入并启动模型
from embedding_service.models import BGEReranker
# 使用默认参数启动本地部署的模型，只需提供模型在本地的路径
# cpu/小规模gpu 建议使用 batch_size=4
BGEReranker.startup("/path/to/local_bge_m3", batch_size=4)
```

上述不带可选参数的调用方式，会使用模型默认参数。`BGEReranker.startup()` 方法完整参数如下：

- `model_name_or_path`: 必须，模型本地路径或 Huggingface 模型名；
- `device`: 可选，默认 None，由模型自行选择设备；
- `query_instruction`: 可选，查询指令，默认 None；
- `passage_instruction`: 可选，段落指令，默认 None；
- `batch_size`: 可选，默认 128，cpu/小规模gpu 建议使用 4；

另外，模型启动时会自动设置上下文窗口 `max_length = 2048`，归一化分值 `normalize = True`，半精度计算 `use_fp16 = False`。

BGE-Reranker 是经过特定训练的“指令感知”模型，在重排序时使用合适的 `query_instruction` 和 `passage_instruction` 参数有可能提升 1~5% 的准确率，但应注意：
1. 使用的指令必须经过严格测试，对同一类任务的查询和段落提供相互对应的指令；
2. 查询和段落的指令应当简洁，说明任务类型，例如: 
   - 查询指令: `"给定用户查询，检索与之相关的法规制度条文："` ；
   - 段落指令: `"这是一个用于信息检索的段落，内容为来自法律法规或规章制度的条文："` ；
3. 指令最后一定要加上冒号；
4. 应当避免为段落提供冗长或过于具体的指令，以免引入偏差。

**2. 重排序**

```python
# 重排序
scores = BGEReranker.rank(query, passages)
print(scores)
```
上述不带可选参数的调用方式，会使用模型启动时设定的参数。`BGEReranker.rank()` 方法完整参数如下：

- `query`: 必须，用户查询文本；
- `passages`: 必须，段落文本或段落文本列表；
- `query_instruction`: 可选，查询指令，默认 None；
- `passage_instruction`: 可选，段落指令，默认 None；
- `batch_size`: 可选，默认 128，cpu/小规模gpu 建议使用 batch_size=4；
- `max_length`: 可选，上下文窗口大小，默认 2048；
- `normalize`: 可选，是否归一化分值，默认 True。

**3. 关闭模型**

由于项目采用类单例工厂模式加载使用模型，当需要加载同一台电脑上的其他本地模型，或需要改变模型启动参数时，应当先关闭当前模型。

```python
# 关闭模型
BGEReranker.shutdown()
```

模型关闭后，可以使用不同的参数再次启动。

### Qwen3-Reranker 系列模型

Qwen3-Reranker 系列模型共有 0.6B、4B、8B 三个尺寸版本，通常建议使用 0.6B 版本即可。

Qwen3-Reranker 系列模型使用类 `Qwen3Reranker` 启动、使用和关闭，用法与 `BGEReranker` 类一致，`startup()` 和 `encode()` 的可选参数较少。

- Qwen3-Reranker 模型不支持 `passage_instruction` 指令，如果希望通过段落指令来指定段落元数据、关注点等，可以将元数据与段落原文拼接，将关注点等排序要求写入 `query_instruction` 中。
- `Qwen3Embedding.rank()` 无 `normalize` 参数，模型强制归一化分值。

关于 `batch_size` 参数的建议同 BGE-Reranker 模型。

### GLM-Rerank 模型

GLM-Rerank 模型的使用方法和其他两个本地部署模型一致，但不支持 `query_instruction`, `batch_size`, `max_length`, `normalize` 参数的自定义值，所有参数均封装在内部不可更改。

由于 GLM 允许此模型最大 50 个并发请求，因此通过内部的合理批次化并行重排，可以获得较高的运行效率，排序耗时相当于 GPU 环境下本地部署的 0.6B 重排序模型。

使用此模型，需要在智谱AI开放平台申请 token 并在 `.env` 中配置为环境变量 `GLM_API_KEY` 或 `GLM_RERANK_API_KEY`，另外还需配置环境变量 `GLM_RERANK_BASE_URL` 为该模型的完整请求 URL，即需要以 `"/rerank"` 结尾。请按照智谱AI官网 SDK 文档所述正确配置。

*注意：调用此模型会产生一定费用。*

## Reranker 返回值

Reranker 模型返回每一个段落与用户查询之间的相关度评分，以浮点数列表形式返回，顺序与调用参数中 `passages` 列表顺序对应。

用户获得返回后，可以通过列表拼接的方式获得每个段落的最终得分，例如：

```python
# 假设有两个 Reranker 模型，分别为 reranker1 和 reranker2
scores1 = reranker1.rank(query, passages)
scores2 = reranker2.rank(query, passages)
# 最终得分为两个模型得分的加权和
final_scores = [s1 * 0.6 + s2 * 0.4 for s1, s2 in zip(scores1, scores2)]
# 根据最终得分对段落进行排序
ranked_passages = [p for _, p in sorted(zip(final_scores, passages), reverse=True)]
print(ranked_passages)
```

