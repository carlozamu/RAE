import os
import warnings

# Suppress ROCm experimental feature warnings
warnings.filterwarnings("ignore", message="Flash Efficient attention")
warnings.filterwarnings("ignore", message="Mem Efficient attention")

# Optional: If you want to explicitly disable the experimental path to be safe
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

import aiohttp
from sentence_transformers import SentenceTransformer


# Singleton instance holder
_EMBEDDER_INSTANCE = None

class LLM:
    def __init__(self, base_url="http://localhost:8000/v1"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.model_name = "google/gemma-3-1b-it" 
        
        # Load Embedding Model ONCE (Global Singleton Pattern)
        global _EMBEDDER_INSTANCE
        if _EMBEDDER_INSTANCE is None:
            print("Loading Embedding Model (SentenceTransformer)...")
            _EMBEDDER_INSTANCE = SentenceTransformer('all-MiniLM-L6-v2')
            print("Embedding Model Loaded.")
        self.embedder = _EMBEDDER_INSTANCE

    def get_embedding(self, text: str) -> list[float]:
        """
        Computes the embedding vector for a given text.
        Synchronous because it runs locally on CPU/GPU via PyTorch, not HTTP.
        """
        return self.embedder.encode(text).tolist()

    async def generate_text(self, user_prompt:str, max_tokens=512, stop=None, temperature=0.2, primer="") -> str:
        """
        Internal method to send raw completion requests with Gemma formatting.
        """
        # 1. Apply the "chat template" format
        formatted_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{primer}"
        
        # 2. Default stop token
        stop_tokens = ["<end_of_turn>"]
        if stop:
            if isinstance(stop, list):
                stop_tokens.extend(stop)
            else:
                stop_tokens.append(stop)

        payload = {
            "model": self.model_name,
            "prompt": formatted_prompt, # Using raw 'prompt', not 'messages'
            "max_tokens": max_tokens,
            "temperature": temperature, # High temp for evolutionary diversity, low tem for logical focus
            "stop": stop_tokens
        }

        try:
            async with aiohttp.ClientSession() as session:
                # Note: Targeting /completions, NOT /chat/completions for compatibility safety
                async with session.post(f"{self.base_url}/completions", headers=self.headers, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    # Parse raw text response
                    return data['choices'][0]['text'].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""
