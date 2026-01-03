import re
import random
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset

class CoTManager:
    """
    Manager for the kaist-ai/CoT-Collection dataset.
    Filters for open-ended answers and provides formatted problems.
    """
    
    YES_NO = {"yes", "no", "true", "false"}
    MC_OPTIONS = {"a", "b", "c", "d", "e"}

    def __init__(self, split: str = "train", cache_dir: Optional[str] = None):
        print(f"Loading CoT-Collection: kaist-ai/CoT-Collection - {split}...")
        try:
            # Note: trust_remote_code=True is required for this dataset as per notebook warning
            self.dataset = load_dataset("kaist-ai/CoT-Collection", split=split, cache_dir=cache_dir, trust_remote_code=True)
            print("CoT-Collection loaded successfully.")
            
            # Pre-filter logic (optional optimization: filter on demand or caching indices)
            # For now, we will filter on the fly or build an index of valid items if dataset is large
            # The notebook filtered first 50k. We can do lazy filtering.
            self.valid_indices = [] # Populated lazily or on init?
            # Let's populate a pool of valid indices to sample from effectively.
            # Scanning 1.8M rows takes time. Let's scan a subset or handle it gracefully.
            self._scan_valid_indices(limit=50000)
            
        except Exception as e:
            print(f"Error loading CoT dataset: {e}")
            self.dataset = None
            self.valid_indices = []

    def _scan_valid_indices(self, limit: int):
        """
        Scans top N items to find valid open-answer questions.
        """
        print(f"Scanning first {limit} generic examples for open-answer candidates...")
        valid = []
        count = 0
        total = min(limit, len(self.dataset)) if self.dataset else 0
        
        for i in range(total):
            item = self.dataset[i]
            target = item.get("target", "")
            if self.is_open_answer_target(target):
                valid.append(i)
            count += 1
            
        self.valid_indices = valid
        print(f"Found {len(self.valid_indices)} valid open-answer examples.")

    def get_batch(self, size: int = 10, random_seed: int = None) -> List[Dict[str, str]]:
        if not self.dataset or not self.valid_indices:
            print("CoT Dataset not ready or empty.")
            return []
            
        if random_seed is not None:
             random.seed(random_seed)
             
        # Sample indices from valid list
        sampled_indices = random.sample(self.valid_indices, min(size, len(self.valid_indices)))
        
        batch = []
        for idx in sampled_indices:
            item = self.dataset[idx]
            source = item["source"]
            target = item["target"]
            rationale = item.get("rationale", "") # CoT dataset has rationale sometimes
            
            prompt = self.build_prompt_cot_zero_shot(source)
            
            batch.append({
                "question": prompt,
                "answer": target,
                "task_type": "cot", # General Chain of Thought
                "rationale": rationale,
                "metadata": {
                    "source": source,
                    "task": item.get("task", "unknown")
                }
            })
            
        return batch

    @staticmethod
    def build_prompt_cot_zero_shot(source: str) -> str:
        return f"""You are a helpful reasoning assistant.

Read the following task and provide the correct answer.
Do NOT explain your reasoning, just output the final answer.

Task:
{source}

Answer:
"""

    @classmethod
    def is_open_answer_target(cls, t: str) -> bool:
        if not t:
            return False
        norm = t.strip().lower()
        # Clean simple punctuation
        norm = re.sub(r"[^a-z0-9 ]", "", norm)
        
        # Filter yes/no/single letters
        if norm in cls.YES_NO or norm in cls.MC_OPTIONS:
            return False
        return True
