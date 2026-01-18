import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    is_torch_npu_available,
)
from tqdm import tqdm, trange


class Qwen3RerankerModel:
    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the "
        "requirements based on the Query and the Instruct provided. "
        'Note that the answer can only be "yes" or "no".'
        "<|im_end|>\n<|im_start|>user\n"
    )
    SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __init__(
        self,
        model_name_or_path: str,
        device: str | None = None,
        instruction: str | None = None,
        batch_size: int = 128,
        **kwargs,
    ) -> None:
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided.")

        self.instruction = (
            instruction or "Given the user query, retrieval the relevant passages"
        )
        self.batch_size = batch_size
        self.max_length = 2048
        self.devices = self.get_local_devices(device)
        self.pool = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side="left",
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            self.token_false_id = self.tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self.tokenizer.convert_tokens_to_ids("yes")
            self.prefix_tokens = self.tokenizer.encode(
                self.PREFIX,
                add_special_tokens=False,
            )
            self.suffix_tokens = self.tokenizer.encode(
                self.SUFFIX,
                add_special_tokens=False,
            )
        except Exception as e:
            self.model = None
            self.tokenizer = None
            raise e

    def shutdown(self):
        if self.model is not None:
            import gc

            if self.pool is not None:
                self.stop_multi_process_pool()
                self.pool = None
            try:
                self.model.to("cpu")
                torch.cuda.empty_cache()
            except:
                pass

            if callable(gc.collect):
                gc.collect()

            self.tokenizer = None
            self.model = None

            self.devices = None
            self.instruction = None
            self.batch_size = None
            self.max_length = None

    def stop_multi_process_pool(self):
        for p in self.pool["processes"]:
            p.terminate()

        for p in self.pool["processes"]:
            p.join()
            p.close()

        self.pool["input"].close()
        self.pool["output"].close()

    def get_local_devices(self, device: str | None) -> list[str]:
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

    # copied and modified from Github Qwen3-Reranker examples
    def format_instruction(self, ins: str, query: str, doc: str) -> str:
        return f"<Instruct>: {ins}\n<Query>: {query}\n<Document>: {doc}"

    # copied and modified from Github Qwen3-Reranker examples
    def process_inputs(self, pairs: list[tuple[str, str]]) -> dict:
        out = self.tokenizer(
            pairs,
            padding=False,
            truncation="longest_first",
            return_attention_mask=False,
            max_length=(
                self.max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
            ),
        )
        for i, ele in enumerate(out["input_ids"]):
            out["input_ids"][i] = self.prefix_tokens + ele + self.suffix_tokens
        out = self.tokenizer.pad(
            out, padding=True, return_tensors="pt", max_length=self.max_length
        )
        for key in out:
            out[key] = out[key].to(self.model.device)
        return out

    # copied and modified from Github Qwen3-Reranker examples
    @torch.no_grad()
    def compute_logits(self, inputs: dict, **kwargs) -> list[float]:
        batch_scores = self.model(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, self.token_true_id]
        false_vector = batch_scores[:, self.token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        scores = batch_scores[:, 1].exp().tolist()
        return scores

    # copied and modified from Github Qwen3-Reranker examples and FlagReranker
    def compute_score_on_single_device(
        self,
        pairs: list[str],
        batch_size: int,
        max_length: int,
        device: str,
        **kwargs,
    ) -> list[float]:
        if device.startswith("cuda"):
            self.model.half()
        self.model.to(device)
        self.model.eval()

        all_scores = []
        for start_index in tqdm(
            range(0, len(pairs), batch_size),
            desc="Compute Scores",
            disable=len(pairs) < batch_size,
        ):
            batch_pairs = pairs[start_index : start_index + batch_size]
            batch_inputs = self.process_inputs(batch_pairs)
            batch_scores = self.compute_logits(batch_inputs)
            all_scores.extend(batch_scores)

        return all_scores

    def compute_score(self, pairs: list[tuple[str, str]], **kwargs) -> list[float]:
        instruction = kwargs.get("instruction", None) or self.instruction
        batch_size = kwargs.get("batch_size", None) or self.batch_size
        max_length = kwargs.get("max_length", None) or self.max_length

        sentence_pairs = [self.format_instruction(instruction, q, p) for q, p in pairs]

        if len(self.devices) > 1:
            if self.pool is None:
                self.pool = self.start_multi_process_pool()
            scores = self.encode_multi_devices(
                sentence_pairs,
                batch_size=batch_size,
                max_length=max_length,
            )
        else:
            scores = self.compute_score_on_single_device(
                sentence_pairs,
                batch_size,
                max_length,
                self.devices[0],
            )

        return scores

    # --- Multi-devices functions ---

    # copied and modified from Github Qwen3-Reranker examples and FlagReranker
    def start_multi_process_pool(self) -> dict:
        import multiprocessing as mp

        self.model.to("cpu")
        self.model.share_memory()
        ctx = mp.get_context("spawn")
        input_queue = ctx.Queue()
        output_queue = ctx.Queue()
        processes = []

        for device in tqdm(self.devices, desc="Initial target device"):
            p = ctx.Process(
                target=Qwen3RerankerModel._encode_multi_process_worker,
                args=(self, device, input_queue, output_queue),
                daemon=True,
            )
            p.start()
            processes.append(p)

        return {"input": input_queue, "output": output_queue, "processes": processes}

    # copied and modified from Github Qwen3-Reranker examples and FlagReranker
    def encode_multi_devices(
        self,
        pairs: list[str],
        **kwargs,
    ) -> list[float]:
        n = len(pairs)
        m = len(self.pool["processes"])
        chunk_size = n // m + int(n % m != 0)

        input_queue = self.pool["input"]
        last_chunk_id = 0
        chunk = []

        for pair in pairs:
            chunk.append(pair)
            if len(chunk) >= chunk_size:
                input_queue.put([last_chunk_id, chunk, kwargs])
                last_chunk_id += 1
                chunk = []

        if len(chunk) > 0:
            input_queue.put([last_chunk_id, chunk, kwargs])
            last_chunk_id += 1

        output_queue = self.pool["output"]
        results_list = sorted(
            [output_queue.get() for _ in trange(last_chunk_id, desc="Chunks")],
            key=lambda x: x[0],
        )
        scores = []
        for list in results_list:
            scores.extend(list[1])

        return scores

    # copied and modified from Github Qwen3-Reranker examples and FlagReranker
    @staticmethod
    def _encode_multi_process_worker(model, device, input_queue, results_queue):
        while True:
            try:
                chunk_id, sentences, kwargs = input_queue.get()
                embeddings = model.compute_score_on_single_device(
                    sentences,
                    device=device,
                    **kwargs,
                )
                results_queue.put([chunk_id, embeddings])
            except:
                break
