from .. import conf, logger
from .abstract_models import AbstractReranker

class BGEReranker(AbstractReranker):
    _model = None

    @classmethod
    def rerank(
        cls,
        query: str,
        passages: list[str],
        batch_size: int | None=None,
    )-> list[float]:
        """
        Rerank passages based on their relevance to the query using BGE model.

        Ensures the model is started up before reranking, otherwise returns
        an empty list.

        Args:
            query (str):
                The query string to compare against passages.
            passages (list[str]):
                A list of passage strings to be reranked.
            batch_size (int | None):
                The batch size for processing passages. Default is None, leading
                to use the batch size set during startup.

        Returns:
            list[float]: A list of relevance scores for each passage.
        """
        # TODO: implement the reranking logic using the BGE model
        pass