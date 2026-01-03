import re
import random
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset
# import torch # Not strictly needed here unless we use tokenizer stuff, but we format strings

class CLUTTRManager:
    RELATIONS = [
        "aunt", "son-in-law", "grandfather", "brother", "sister",
        "father", "mother", "grandmother", "uncle", "daughter-in-law",
        "grandson", "granddaughter", "father-in-law", "mother-in-law",
        "nephew", "son", "daughter", "niece", "husband", "wife",
        "sister-in-law"
    ]

    SYNONYM_MAP = {
        "grandma": "grandmother",
        "grandpa": "grandfather",
        "mom": "mother",
        "dad": "father",
        "sis": "sister",
        "bro": "brother",
        "soninlaw": "son-in-law",
        "daughterinlaw": "daughter-in-law",
    }

    def __init__(self, split_config: str = "gen_train234_test2to10", cache_dir: Optional[str] = None):
        """
        Initialize the CLUTTR dataset loader.
        :param split_config: The configuration name for CLUTTR (e.g. "gen_train234_test2to10")
        """
        print(f"Loading CLUTTR dataset: CLUTRR/v1 - {split_config}...")
        try:
            self.dataset = load_dataset("CLUTRR/v1", split_config, cache_dir=cache_dir)
            print("CLUTTR dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading CLUTTR dataset: {e}")
            self.dataset = None

    def get_batch(self, split: str = "train", size: int = 10, random_seed: int = None) -> List[Dict[str, str]]:
        """
        Returns a batch of formatted problems.
        Each problem is a dict: {'question': str, 'answer': str, 'metadata': dict}
        """
        if self.dataset is None or split not in self.dataset:
            print(f"Dataset not valid or split '{split}' not found.")
            return []
        
        data_split = self.dataset[split]
        total_len = len(data_split)
        
        if random_seed is not None:
             random.seed(random_seed)
        
        # Select random indices
        indices = random.sample(range(total_len), min(size, total_len))
        
        batch = []
        for idx in indices:
            item = data_split[idx]
            story = item["story"]
            query = item["query"]
            target = item["target_text"]
            
            prompt = self.build_prompt_clutrr_zero_shot(story, query)
            
            batch.append({
                "question": prompt,
                "answer": target,
                "task_type": "cluttr",
                "metadata": {
                    "story": story,
                    "query": query,
                    "id": item.get("id", None)
                }
            })
            
        return batch

    @staticmethod
    def build_prompt_clutrr_zero_shot(story: str, query: str) -> str:
        # If query is a tuple string representation like "('Ashley', 'Nicholas')", clean it up for display? 
        # The notebook passes it directly into {query} in the f-string.
        # Let's inspect the query format. In the notebook output: ("Ashley", "Nicholas")
        
        return f"""You will be given a short story about family members and their relationships.
From this story, infer the exact family relationship between the two people in the query.

Story:
{story}

Query:
What is the family relationship of {query}?

Answer using a single English kinship noun like "aunt", "father", "niece", "grandmother", etc.
Return ONLY the single word relationship, nothing else.
"""

    @classmethod
    def normalize_text_simple(cls, text: str) -> str:
        if not text:
            return ""
        text = text.lower()
        text = re.sub(r"[^a-z\- ]", " ", text)
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    @classmethod
    def map_to_relation(cls, text: str) -> str:
        """
        Extracts the relationship from the model's output using strict mapping and synonyms.
        """
        if not text:
            return ""

        norm_full = cls.normalize_text_simple(text)
        tokens = norm_full.split()

        # 1. Exact match in tokens
        for tok in tokens:
            if tok in cls.RELATIONS:
                return tok

        # 2. Synonym match
        for tok in tokens:
            if tok in cls.SYNONYM_MAP:
                return cls.SYNONYM_MAP[tok]

        # 3. Joined match (e.g. "soninlaw")
        joined = norm_full.replace(" ", "")
        if joined in cls.SYNONYM_MAP:
            return cls.SYNONYM_MAP[joined]
        
        # Extra check: if the joined string IS a relation (e.g. "son-in-law" became "soninlaw" in norm? No, norm keeps hyphens)
        # But if user output "son in law", norm is "son in law", joined is "soninlaw".
        # We need to check if joined matches a relation with hyphens removed.
        
        clean_relations = {r.replace("-", ""): r for r in cls.RELATIONS}
        if joined in clean_relations:
            return clean_relations[joined]

        return ""
