from .. import logger
from .abstract_models import AbstractReranker

class Qwen3Reranker(AbstractReranker):
    _tokenizer = None
    _model = None
    _pool = None
    _devices = None
    _query_instruction = None
    _batch_size = None
    _max_length = None

    _PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the "
        "requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
        "<|im_end|>\n<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    @classmethod
    def rank(
        cls,
        query: str,
        passages: str | list[str],
        query_instruction: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        **kwargs,
    )-> list[float]:
        """
        Rerank passages based on their relevance to the query using BGE model.

        Ensures the model is started up before reranking, otherwise returns
        an empty list.

        Qwen3-Reranker models confirm to normalize the final scores. It does
        not use passage instruction for reranker.

        Args:
            query (str):
                The query string to compare against passages.
            passages (list[str]):
                A list of passage strings to be reranked.
            query_instruction: (str | None):
                Instruction for queries.
            batch_size (int | None):
                The batch size for processing passages. Default is None.
            max_length (int | None):
                The max length of context, default 3072.

        Returns:
            list[float]: A list of relevance scores for each passage.
        """
        if not query:
            logger.warning("Query must be provided.")
            return []
        if not passages:
            logger.warning("Passages must be provided.")
            return []
        if cls._model is None:
            logger.warning("Model is not started.")
            return []

        pairs = [query, passages] if isinstance(passages, str) else [
            [query, p] for p in passages
        ]

        return []

#         if query_instruction:
#             cls._model.query_instruction_for_rerank = query_instruction
#         if passage_instruction:
#             cls._model.passage_instruction_for_rerank = passage_instruction
# 
#         return cls._model.compute_score(
#             pairs,
#             batch_size=batch_size,
#             max_length=max_length,
#             normalize=normalize,
#         )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str,
        device: str | None = None,
        query_instruction: str | None = None,
        batch_size: int = 128,
        **kwargs,
    )-> None:
        """
        Initialize, load and warm-up BGE-Reranker-v2-m3 model.

        The batch size and instructions are set during startup and will be
        used as defaults. They can be overridden while compute scores.

        Qwen3-Reranker models confirm to normalize the final scores. It does
        not use passage instruction for reranker.

        Other default parameters including:
        - max_length = 3072

        Args:
            model_name_or_path (str):
                Path to the model. None or empty will cause a ValueError.
            device (str | None):
                Device to run the model on.
            query_instruction: (str | None):
                Instruction for queries.
            batch_size: (int):
                batch size, default is 128.
        """
        if cls._model is not None:
            return
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided.")

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            is_torch_npu_available,
        )

        cls._query_instruction = (
            query_instruction or 
            "Given the user query, retrieval the relevant passages"
        )
        cls._batch_size = batch_size
        cls._max_length = 3072
        cls._devices = cls.get_local_devices(device)
        cls._pool = None

        try:
            cls._tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side='left',
            )
            cls._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            cls.token_false_id = cls._tokenizer.convert_tokens_to_ids("no")
            cls.token_true_id = cls._tokenizer.convert_tokens_to_ids("yes")
            cls.prefix_tokens = cls._tokenizer.encode(
                cls._PREFIX,
                add_special_tokens=False,
            )
            cls.suffix_tokens = cls._tokenizer.encode(
                cls._SUFFIX,
                add_special_tokens=False,
            )
            logger.info(f"{model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {model_name_or_path} failed: {e}")
            cls._model = None
            return

    @classmethod
    def shutdown(cls):
        if cls._model is not None:
            cls._tokenizer = None
            cls._model = None
            cls._pool = None    # TODO
            cls._devices = None
            cls._query_instruction = None
            cls._batch_size = None
            cls._max_length = None
            logger.info("Qwen3-Reranker model shutdown.")

    @staticmethod
    def get_local_devices(device: str | None)-> list[str]:
        import torch
        from transformers import is_torch_npu_available

        if not device:
            if torch.cuda.is_available():
                return [f"cuda:{i}" for i in range(torch.cuda.device_count())]
            elif is_torch_npu_available():
                return [f"npu:{i}" for i in range(torch.npu.device_count())]
            elif hasattr(torch, "musa") and torch.musa.is_available():
                return [f"musa:{i}" for i in range(torch.musa.device_count())]
            elif torch.backends.mps.is_available():
                return ["mps"]
            else:
                return ["cpu"]
        else:
            return [device]


