class Qwen3RerankerModel:
    PREFIX = (
        "<|im_start|>system\nJudge whether the Document meets the "
        "requirements based on the Query and the Instruct provided. "
        "Note that the answer can only be \"yes\" or \"no\"."
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
    )-> None:
        if not model_name_or_path:
            raise ValueError("model_name_or_path must be provided.")

        import torch
        from transformers import (
            AutoTokenizer,
            AutoModelForCausalLM,
            is_torch_npu_available,
        )

        self.instruction = (
            instruction or 
            "Given the user query, retrieval the relevant passages"
        )
        self.batch_size = batch_size
        self.max_length = 2048
        self.devices = get_local_devices(device)
        self.pool = None

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                padding_side='left',
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=True,
                dtype=torch.float32,
            )
            self.token_false_id = self._tokenizer.convert_tokens_to_ids("no")
            self.token_true_id = self._tokenizer.convert_tokens_to_ids("yes")
            self.prefix_tokens = self._tokenizer.encode(
                self.PREFIX,
                add_special_tokens=False,
            )
            self.suffix_tokens = self._tokenizer.encode(
                self.SUFFIX,
                add_special_tokens=False,
            )
        except Exception as e:
            self._model = None
            self._tokenizer = None
            raise e

    # --- Inner functions ---

    @staticmethod
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

