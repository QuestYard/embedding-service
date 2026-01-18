from .. import logger
from .abstract_models import AbstractReranker


class BGEReranker(AbstractReranker):
    _model = None

    @classmethod
    def rank(
        cls,
        query: str,
        passages: str | list[str],
        query_instruction: str | None = None,
        passage_instruction: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        normalize: bool | None = None,
        **kwargs,
    ) -> list[float]:
        """
        Rerank passages based on their relevance to the query using BGE model.

        Ensures the model is started up before reranking, otherwise returns
        an empty list.

        Args:
            query (str):
                The query string to compare against passages.
            passages (list[str]):
                A list of passage strings to be reranked.
            query_instruction: (str | None):
                Instruction for queries.
            passage_instruction: (str | None):
                Instruction for passages.
            batch_size (int | None):
                The batch size for processing passages. Default is None.
            max_length (int | None):
                The max length of context, default 3072.
            normalize (bool | None):
                Whether to normalize the scores, default True.

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

        pairs = (
            [query, passages]
            if isinstance(passages, str)
            else [[query, p] for p in passages]
        )

        if query_instruction:
            cls._model.query_instruction_for_rerank = query_instruction
        if passage_instruction:
            cls._model.passage_instruction_for_rerank = passage_instruction

        return cls._model.compute_score(
            pairs,
            batch_size=batch_size,
            max_length=max_length,
            normalize=normalize,
        )

    @classmethod
    def startup(
        cls,
        model_name_or_path: str,
        device: str | None = None,
        query_instruction: str | None = None,
        passage_instruction: str | None = None,
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        """
        Initialize, load and warm-up BGE-Reranker-v2-m3 model.

        The batch size and instructions are set during startup and will be
        used as defaults. They can be overridden while compute scores.

        Other default parameters including:
        - normalize = True, can be overridden.
        - max_length = 2048.
        - use_fp16 = False, fixed.

        Args:
            model_name_or_path (str):
                Path to the model. None or empty will cause a ValueError.
            device (str | None):
                Device to run the model on.
            query_instruction: (str | None):
                Instruction for queries.
            passage_instruction: (str | None):
                Instruction for passages.
            batch_size: (int):
                BGE-Reranker-v2-m3 default batch size, 128.
        """
        if cls._model is not None:
            return

        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided.")

        from FlagEmbedding import FlagReranker

        try:
            cls._model = FlagReranker(
                model_name_or_path.strip(),
                devices=device,
                query_instruction_for_rerank=query_instruction,
                passage_instruction_for_rerank=passage_instruction,
                batch_size=batch_size,
                normalize=True,
                use_fp16=False,
                max_length=2048,
            )
            logger.info(f"{model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {model_name_or_path} failed: {e}")
            cls._model = None
            return

        try:
            _ = cls._model.compute_score(["query", "passage"])
            logger.info(f"{model_name_or_path} warmed-up.")
        except Exception as e:
            logger.error(f"Warming-up {model_name_or_path} failed: {e}")
            return

    @classmethod
    def shutdown(cls):
        if cls._model is not None:
            cls._model = None
            logger.info("BGE-Reranker-v2-m3 model shutdown.")
