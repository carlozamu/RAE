import asyncio
import datetime
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
        self.semaphore = asyncio.Semaphore(50)

        # Load Embedding Model ONCE (Global Singleton Pattern)
        global _EMBEDDER_INSTANCE
        embedding_model_name = 'BAAI/bge-base-en-v1.5'
        if _EMBEDDER_INSTANCE is None:
            print("Loading Embedding Model from SentenceTransformer...")
            _EMBEDDER_INSTANCE = SentenceTransformer(embedding_model_name, device='cpu')
            print(f"{embedding_model_name} Embedding Model Loaded.")
        self.embedder = _EMBEDDER_INSTANCE

    def get_embedding(self, text: str) -> list[float]:
        """
        Computes the embedding vector for a given text.
        Synchronous because it runs locally on CPU/GPU via PyTorch, not HTTP.
        """
        return self.embedder.encode(text).tolist()

    async def generate_text(self, user_prompt:str, temperature:float, max_tokens=512, stop=None) -> str:
        async with self.semaphore:
            # 1. Format the prompt (Gemma specific)        
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
                    "prompt": user_prompt,
                    "stream": False,
                    "raw": True,
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
                    "prompt": user_prompt,
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
                            answer = data.get('response', '').strip()
                        else:
                            answer = data['choices'][0]['text'].strip()

                        # Ensure the logs directory exists
                        log_file = "Utils/Logs/server_logs.md"
                        os.makedirs(os.path.dirname(log_file), exist_ok=True)

                        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        
                        # Construct the formatted Markdown entry
                        log_entry = f"""
**⚙️ Parameters:**
- `Model`: {self.model_name}
- `Temperature`: {temperature}
- `Max Tokens`: {max_tokens}
- `Timestamp`: {timestamp}
---------------------------------------------
**📥 Query:**
```text
{user_prompt.strip()}
```
---------------------------------------------
**📤 Response:**
```text
{answer.strip()}
```
---------------------------------------------"""
                        
                        # Append the message to the markdown file
                        with open(log_file, "a", encoding="utf-8") as f:
                            # We strip leading newlines to avoid weird markdown formatting gaps, 
                            # but keep the newline at the end for the next log.
                            f.write(log_entry + "\n\n")
                        
                        return answer
                    
            except Exception as e:
                print(f"LLM Error: {e}")
                return ""
        