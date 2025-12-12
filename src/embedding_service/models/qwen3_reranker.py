from .. import logger
from .abstract_models import AbstractReranker

PREFIX = (
    "<|im_start|>system\nJudge whether the Document meets the "
    "requirements based on the Query and the Instruct provided. "
    "Note that the answer can only be \"yes\" or \"no\"."
    "<|im_end|>\n<|im_start|>user\n"
)
SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = None
suffix_tokens = None
token_true_id = None
token_false_id = None

class Qwen3Reranker(AbstractReranker):
    _tokenizer = None
    _model = None
    _pool = None
    _devices = None
    _query_instruction = None
    _batch_size = None
    _max_length = None

    # _PREFIX = (
    #     "<|im_start|>system\nJudge whether the Document meets the "
    #     "requirements based on the Query and the Instruct provided. "
    #     "Note that the answer can only be \"yes\" or \"no\"."
    #     "<|im_end|>\n<|im_start|>user\n"
    # )
    # _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

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

        pairs = [(query, passages)] if isinstance(passages, str) else [
            (query, p) for p in passages
        ]
        pairs = [
            format_instruction(
                query_instruction or cls._query_instruction, q, p
            ) for q, p in pairs
        ]
        scores = compute_scores(
            pairs,
            batch_size or cls._batch_size,
            max_length or cls._max_length,
            cls._devices[0],
            model = cls._model,
            tokenizer = cls._tokenizer,
            # token_false_id = cls.token_false_id,
            # token_true_id = cls.token_true_id,
            # prefix_tokens = cls.prefix_tokens,
            # suffix_tokens = cls.suffix_tokens,
        )

        return scores

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
        - max_length = 2048

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
        cls._max_length = 2048
        cls._devices = get_local_devices(device)
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
            # cls.token_false_id = cls._tokenizer.convert_tokens_to_ids("no")
            # cls.token_true_id = cls._tokenizer.convert_tokens_to_ids("yes")
            # cls.prefix_tokens = cls._tokenizer.encode(
            #     cls._PREFIX,
            #     add_special_tokens=False,
            # )
            # cls.suffix_tokens = cls._tokenizer.encode(
            #     cls._SUFFIX,
            #     add_special_tokens=False,
            # )
            token_false_id = cls._tokenizer.convert_tokens_to_ids("no")
            token_true_id = cls._tokenizer.convert_tokens_to_ids("yes")
            prefix_tokens = cls._tokenizer.encode(
                PREFIX,
                add_special_tokens=False,
            )
            suffix_tokens = cls._tokenizer.encode(
                SUFFIX,
                add_special_tokens=False,
            )
            logger.info(f"{model_name_or_path} loaded.")
        except Exception as e:
            logger.error(f"Loading {model_name_or_path} failed: {e}")
            cls._model = None
            cls._tokenizer = None
            return

    @classmethod
    def shutdown(cls):
        if cls._model is not None:
            import gc
            import torch
            if cls._pool is not None:
                cls.stop_multi_process_pool()
                cls._pool = None
            try:
                cls._model.to("cpu")
                torch.cuda.empty_cache()
            except:
                pass

            if callable(gc.collect):
                gc.collect()

            cls._tokenizer = None
            cls._model = None

            cls._devices = None
            cls._query_instruction = None
            cls._batch_size = None
            cls._max_length = None
            logger.info("Qwen3-Reranker model shutdown.")

    @classmethod
    def stop_multi_process_pool(cls):
        for p in cls._pool["processes"]:
            p.terminate()

        for p in cls._pool["processes"]:
            p.join()
            p.close()

        cls._pool["input"].close()
        cls._pool["output"].close()

    @classmethod
    def test_mp(cls):
        cls._pool = start_multi_process_pool()

# --- Inner functions ---

def get_local_devices(device: str | None=None)-> list[str]:
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

def format_instruction(ins, query, doc):
    return f"<Instruct>: {ins}\n<Query>: {query}\n<Document>: {doc}"

def compute_scores(pairs, batch_size, max_length, device, **kwargs):
    model = kwargs.get("model", None)
    tokenizer = kwargs.get("tokenizer", None)
    if device.startswith("cuda"):
        model.half()
    model.to(device)
    model.eval()
    tokenizer = kwargs.get("tokenizer", None)

    # token_true_id = kwargs.get("token_true_id", None)
    # token_false_id = kwargs.get("token_false_id", None)
    # prefix_tokens = kwargs.get("prefix_tokens", None)
    # suffix_tokens = kwargs.get("suffix_tokens", None)

    from tqdm import tqdm
    import torch

    with torch.no_grad():
        all_scores = []
        for start_index in tqdm(
            range(0, len(pairs), batch_size),
            desc="Compute Scores",
            disable=len(pairs) < batch_size,
        ):
            batch_pairs = pairs[start_index:start_index + batch_size]
            batch_inputs = process_inputs(
                batch_pairs,
                max_length,
                model.device,
                **kwargs,
            )
            # process inputs
            batch_inputs = tokenizer(
                batch_pairs,
                padding=False,
                truncation='longest_first',
                return_attention_mask=False,
                max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
            )
            for i, ele in enumerate(batch_inputs['input_ids']):
                batch_inputs['input_ids'][i] = (
                    prefix_tokens + ele + suffix_tokens
                )
            batch_inputs = tokenizer.pad(
                batch_inputs,
                padding=True,
                return_tensors="pt",
                max_length=max_length,
            )
            for key in batch_inputs:
                batch_inputs[key] = batch_inputs[key].to(model.device)
            # copied from Qwen3Reranker Github example
            batch_scores = model(**batch_inputs).logits[:, -1, :]
            true_vector = batch_scores[:, token_true_id]
            false_vector = batch_scores[:, token_false_id]
            batch_scores = torch.stack([false_vector, true_vector], dim=1)
            batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
            scores = batch_scores[:, 1].exp().tolist()
            all_scores.extend(scores)

    return all_scores

# codes below are copied from SentenceTransformer.py and slightly modified.
def start_multi_process_pool(devices, model, tokenizer):
    logger.info(f"Start multi-process pool on devices: {devices}")

    import multiprocessing as mp
    from tqdm import tqdm

    model.to("cpu")
    model.share_memory()
    ctx = mp.get_context("spawn")
    input_queue = ctx.Queue()
    output_queue = ctx.Queue()
    processes = []

    for device_id in tqdm(devices, desc='initial target device'):
        p = ctx.Process(
            target=_encode_multi_process_worker,
            args=(model, tokenizer, device_id, input_queue, output_queue),
            daemon=True,
        )
        p.start()
        processes.append(p)

    return {
        "input": input_queue,
        "output": output_queue,
        "processes": processes
    }

def _encode_multi_process_worker(
    model,
    tokenizer,
    target_device,
    input_queue,
    results_queue,
) -> None:
    logger.warning(f"{model = }")
    logger.warning(f"{tokenizer = }")
    logger.warning(f"{target_device = }")
    while True:
        try:
            pass
            # chunk_id, sentences, kwargs = (
            #     input_queue.get()
            # )
            # embeddings = model.compute_score_single_gpu(
            #     sentences,
            #     device=target_device,
            #     **kwargs
            # )

            # results_queue.put([chunk_id, embeddings])
        except:
            break

