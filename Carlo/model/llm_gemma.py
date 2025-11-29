import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from Carlo.model.llm_client import LLMClient


class Gemma3Client(LLMClient):
    """
    LLM client for Gemma 3 270M (HF). Loads once, exposes `generate(prompt)`.
    By default avoids `device_map="auto"` to not require `accelerate`.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-270m-it",
        max_new_tokens: int = 64,
        do_sample: bool = False,
        device: str | None = None,
        use_device_map: bool = False,
    ):
        self.model_id = model_id
        self.max_new_tokens = max_new_tokens
        self.do_sample = do_sample

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        # Pick device
        if device:
            torch_device = torch.device(device)
        else:
            torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Avoid accelerate requirement unless explicitly requested
        model_kwargs = {
            "dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        }
        if use_device_map:
            model_kwargs["device_map"] = "auto"

        self.model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
        if not use_device_map:
            self.model.to(torch_device)

        # Clean unsupported generation flags to avoid warnings.
        if hasattr(self.model, "generation_config"):
            try:
                self.model.generation_config.top_p = None
                self.model.generation_config.top_k = None
            except Exception:
                pass

        self.model.eval()

    def generate(self, prompt: str) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.eos_token_id,
                # Do not pass top_p/top_k to avoid warnings on models that ignore them.
            )
        gen_tokens = out[0][inputs.input_ids.shape[1]:]
        return self.tokenizer.decode(gen_tokens, skip_special_tokens=True).strip()
