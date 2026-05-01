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
    def __init__(self, model_name="google/gemma-3-1b-it", base_url="http://localhost:8000"):
        self.base_url = base_url
        self.headers = {"Content-Type": "application/json"}
        self.model_name = model_name

        # Load Embedding Model ONCE (Global Singleton Pattern)
        global _EMBEDDER_INSTANCE
        if _EMBEDDER_INSTANCE is None:
            print("Loading Embedding Model (SentenceTransformer)...")
            _EMBEDDER_INSTANCE = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
            print("Embedding Model Loaded.")
        self.embedder = _EMBEDDER_INSTANCE

    def get_embedding(self, text: str) -> list[float]:
        """
        Computes the embedding vector for a given text.
        Synchronous because it runs locally on CPU/GPU via PyTorch, not HTTP.
        """
        return self.embedder.encode(text).tolist()

    async def generate_text(self, user_prompt:str, max_tokens=512, stop=None, temperature=0.2, primer="") -> str:
        # 1. Format the prompt (Gemma specific)
        formatted_prompt = f"<start_of_turn>user\n{user_prompt}<end_of_turn>\n<start_of_turn>model\n{primer}"
        
        stop_tokens = ["<end_of_turn>"]
        if stop:
            stop_tokens.extend(stop if isinstance(stop, list) else [stop])

        # 2. Determine if we are talking to Ollama or vLLM
        is_ollama = "11434" in self.base_url
        
        # 3. Adjust Payload
        if is_ollama:
            # Ollama Native API format
            endpoint = f"{self.base_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": formatted_prompt,
                "stream": False,
                "options": {
                    "num_predict": max_tokens, # Ollama's version of max_tokens
                    "temperature": temperature,
                    "stop": stop_tokens,
                    "num_ctx": 4096 # Essential to keep consistent with vLLM
                }
            }
        else:
            # vLLM / OpenAI format
            endpoint = f"{self.base_url}/v1/completions"
            payload = {
                "model": self.model_name,
                "prompt": formatted_prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "stop": stop_tokens
            }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(endpoint, headers=self.headers, json=payload) as resp:
                    resp.raise_for_status()
                    data = await resp.json()
                    
                    # 4. Handle Different Response Structures
                    if is_ollama:
                        return data.get('response', '').strip()
                    else:
                        return data['choices'][0]['text'].strip()
        except Exception as e:
            print(f"LLM Error: {e}")
            return ""