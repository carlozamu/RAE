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
            task_name = item.get("task_name", "")
            
            # Extract the reasoning length (num2) from "task_[num1].[num2]"
            try:
                reasoning_length = int(task_name.split(".")[1])
            except (IndexError, ValueError):
                reasoning_length = 0 # Fallback if data is malformed
            
            prompt = self.build_prompt_clutrr(story, query)
            
            batch.append({
                "question": prompt,
                "answer": target,
                "task_type": "cluttr",
                "metadata": {
                    "story": story,
                    "query": query,
                    "task_name": task_name,
                    "reasoning_length": reasoning_length,
                    "split_origin": "train",
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
            task_name = item.get("task_name", "")
                
            # Extract the reasoning length (num2) from "task_[num1].[num2]"
            try:
                reasoning_length = int(task_name.split(".")[1])
            except (IndexError, ValueError):
                reasoning_length = 0 # Fallback if data is malformed
            
            # Using your highly optimized zero-shot list prompt
            prompt = self.build_prompt_clutrr_baseline(story, query)
            
            batch.append({
                "question": prompt,
                "answer": target,
                "task_type": "cluttr",
                "metadata": {
                    "story": story,
                    "query": query,
                    "reasoning_length": reasoning_length,
                    "task_name": task_name,
                    "split_origin": split,
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
                
                prompt = self.build_prompt_clutrr_baseline(story, query)
                
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
    def build_prompt_clutrr_baseline(story: str, query: str) -> str:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # 1. Place the strict constraint at the absolute bottom (Recency Bias)
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

        baseline_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        
        return baseline_prompt
    
    @staticmethod
    def build_prompt_clutrr_few_shots(story: str, query: str) -> str:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # 1. Compress options to save context window and improve attention gravity
        options = "aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece, husband, wife, sister-in-law"

        # 2. Construct true multi-turn few-shot history
        few_shot_prompt = f"""<start_of_turn>user
Possible relationships: [{options}]

Story: [Ashley]'s daughter, [Lillian], asked her mom to read her a story. [Nicholas]'s sister [Lillian] asked him for some help planting her garden.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Nicholas is Ashley's?<end_of_turn>
<start_of_turn>model
son<end_of_turn>
<start_of_turn>user
Story: [Wayne] was looking forward to his wife [Nancy] coming back home. She was away for the weekend with her daughter [Lorraine].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Lorraine is Wayne's?<end_of_turn>
<start_of_turn>model
daughter<end_of_turn>
<start_of_turn>user
Story: [Angelica] was mad at her brother [Ronald], because [Ronald] had called her fat. [Marie] made her son [Ronald] and her daughter [Michelle] a cake to take to [Michelle]'s father [Robert] because it was his birthday.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Robert is Angelica's?<end_of_turn>
<start_of_turn>model
father<end_of_turn>
<start_of_turn>user
Story: {story}
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {name2} is {name1}'s?<end_of_turn>
<start_of_turn>model
"""
        
        return few_shot_prompt
    
    @staticmethod
    def build_prompt_clutrr(story: str, query: str) -> str:
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

Understand the family relationship between {name2} and {name1}, and to describe it through only one kinship word from the posible answers, to correctly answer the question: {name2} is {name1}'s?"""
        
        return prompt

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
        Penalizes 'shotgunning': if multiple distinct kinship terms are found, it returns "".
        """
        if not text:
            return ""

        norm_full = cls.normalize_text_simple(text)
        tokens = norm_full.split()

        # 1. Edge Case Protection: Perfect joined match (e.g., text is exactly "son in law")
        # We do this FIRST so the word "son" doesn't trigger a false positive before the 
        # joined string "soninlaw" is evaluated.
        joined = norm_full.replace(" ", "")
        if joined in cls.SYNONYM_MAP:
            return cls.SYNONYM_MAP[joined]
            
        clean_relations = {r.replace("-", ""): r for r in cls.RELATIONS}
        if joined in clean_relations:
            return clean_relations[joined]

        # 2. Collect all distinct relations found in the text
        found_relations = set()

        for tok in tokens:
            if tok in cls.RELATIONS:
                found_relations.add(tok)
            elif tok in cls.SYNONYM_MAP:
                # Add the canonical relation, not the synonym
                found_relations.add(cls.SYNONYM_MAP[tok])

        # 3. Strict Evaluation
        # If exactly ONE unique relationship type was found, return it.
        if len(found_relations) == 1:
            return list(found_relations)[0]
            
        # If 0 were found, or > 1 were found (the model is yapping/guessing), fail it.
        return ""