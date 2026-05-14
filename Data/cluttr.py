import re
import random
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset

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

    def get_batch(self, split: str = "train", batch_size: int = 20, random_seed: int = None) -> List[Dict[str, str]]:
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
        indices = random.sample(range(total_len), min(batch_size, total_len))
        
        batch = []
        for idx in indices:
            item = data_split[idx]
            story = item["story"]
            query = item["query"]
            target = item["target_text"]
            
            sys_instr, prompt, primer = self.build_prompt_clutrr(story, query)
            
            batch.append({
                "system_instructions": sys_instr,
                "question": prompt,
                "primer": primer,
                "answer": target,
                "task_type": "cluttr",
                "metadata": {
                    "story": story,
                    "query": query,
                    "id": item.get("id", None)
                }
            })
            
        return batch
    
    def get_full_split(self, split: str = "test") -> list[dict]:
        """
        Returns every single problem in the specified dataset split.
        Used for infallible baselines and final evolutionary evaluation.
        """
        if self.dataset is None or split not in self.dataset:
            print(f"Dataset not valid or split '{split}' not found.")
            return []
        
        data_split = self.dataset[split]
        total_len = len(data_split)
        print(f"Loading all {total_len} problems from the '{split}' split...")
        
        batch = []
        for idx in range(total_len):
            item = data_split[idx]
            story = item["story"]
            query = item["query"]
            target = item["target_text"]
            
            # Using your highly optimized zero-shot list prompt
            prompt = self.build_prompt_clutrr_baseline_no_primer(story, query)
            
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

    def get_entire_dataset_stratified(self) -> list[dict]:
        """
        Returns every single problem across ALL splits (train, validation, test).
        Extracts the reasoning hop length from 'task_name' for stratified analysis.
        """
        if self.dataset is None:
            print("Dataset not loaded.")
            return []
        
        batch = []
        # Iterate over train, test, and validation (if it exists)
        for split_name in self.dataset.keys():
            data_split = self.dataset[split_name]
            
            for item in data_split:
                story = item["story"]
                query = item["query"]
                target = item["target_text"]
                task_name = item.get("task_name", "")
                
                # Extract the reasoning length (num2) from "task_[num1].[num2]"
                try:
                    reasoning_length = int(task_name.split(".")[1])
                except (IndexError, ValueError):
                    reasoning_length = 0 # Fallback if data is malformed
                
                prompt = self.build_prompt_clutrr_baseline_no_primer(story, query)
                
                batch.append({
                    "question": prompt,
                    "answer": target,
                    "task_type": "cluttr",
                    "metadata": {
                        "story": story,
                        "query": query,
                        "task_name": task_name,
                        "reasoning_length": reasoning_length,
                        "split_origin": split_name
                    }
                })
                
        print(f"Loaded a total of {len(batch)} problems across all splits.")
        return batch
    
    @staticmethod
    def build_prompt_clutrr_baseline(story: str, query: str) -> tuple[str, str]:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # OPTIMIZATION: Naming the entities directly in the task line
        # to maximize attention gravity right before generation.
        prompt = f"""Story:
{story}

Possible answers:
- aunt
- son-in-law
- grandfather
- brother
- sister
- father
- mother
- grandmother
- uncle
- daughter-in-law
- grandson
- granddaughter
- father-in-law
- mother-in-law
- nephew
- son
- daughter
- niece
- husband
- wife
- sister-in-law


Task: State only the one kinship word (from the posible answers) that describes the family relationship between {name2} and {name1}. {name2} is {name1}'s?"""


        baseline_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
        
        return baseline_prompt
    
    @staticmethod
    def build_prompt_clutrr_baseline_no_primer(story: str, query: str) -> tuple[str, str]:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # OPTIMIZATION: Naming the entities directly in the task line
        # to maximize attention gravity right before generation.
        prompt = f"""Story:
{story}

Task: Understand and state the one kinship word that describes the family relationship between {name2} and {name1}. {name2} is {name1}'s?"""

        baseline_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n"
        
        return baseline_prompt

    @staticmethod
    def build_prompt_clutrr(story: str, query: str) -> tuple[str, str]:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        system_instructions = "You are a logical reasoning AI."

        # OPTIMIZATION: Naming the entities directly in the task line
        # to maximize attention gravity right before generation.
        gen_zero_prompt = f"""Story:
{story}

Task: Trace the family lineage step-by-step to find the exact kinship noun connecting {name2} to {name1}."""
        
        # The primer traps the generation
        primer = f"Answer: {name2} is {name1}'s "
        
        return system_instructions, gen_zero_prompt, primer

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
