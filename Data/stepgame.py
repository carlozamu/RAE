import json
import re
import random
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset
import os
from collections import defaultdict

from Data.clutrr import CLUTTRManager

class StepGameManager(CLUTTRManager):
    RELATIONS = [
        "above", "below", "left", "lower-left", "lower-right",
        "overlap", "right", "upper-left", "upper-right"
    ]
    SYNONYM_MAP = {
        "lower-left": "lower left",
        "upper-left": "upper left",
        "upper-right": "upper right",
        "lower-right": "lower right"
    }

    @staticmethod
    def extract_query_names(query: str) -> Tuple[str, str]:
        if not query:
            return "agent A", "agent B"

        agent_matches = re.findall(r"agent\s+([A-Za-z])", query, flags=re.IGNORECASE)
        if len(agent_matches) >= 2:
            return f"agent {agent_matches[0].upper()}", f"agent {agent_matches[1].upper()}"

        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
            return name1, name2
        except ValueError:
            return "agent A", "agent B"

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize the StepGame dataset loader.
        """
        print("Loading StepGame dataset...")
        try:
            self.dataset = load_dataset("ZhengyanShi/StepGame", cache_dir=cache_dir)
            print("StepGame dataset loaded successfully.")
        except Exception as e:
            print(f"Error loading StepGame dataset: {e}")
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
            query = item["question"]
            target = item["label"]
            task_name = item.get("k_hop", "")

            try:
                reasoning_length = int(task_name)
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
            target = item["label"]
            task_complexity = item.get("k_hop", "")
                
            # Extract the reasoning length (num2) from "task_[num1].[num2]"
            try:
                reasoning_length = int(task_complexity)
            except (IndexError, ValueError):
                reasoning_length = 0 # Fallback if data is malformed
            
            # Using your highly optimized zero-shot list prompt
            prompt = self.build_prompt_stepgame_baseline(story, query)
            
            batch.append({
                "question": prompt,
                "answer": target,
                "task_type": "stepgame",
                "metadata": {
                    "story": story,
                    "query": query,
                    "reasoning_length": reasoning_length,
                    "task_name": task_complexity,
                    "split_origin": split,
                    "id": item.get("id", None)
                }
            })
            
        return batch

    def get_entire_dataset_stratified(self, prompt_function: callable, examples:str = None) -> list[dict]:
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
                query = item["question"]
                target = item["label"]
                task_complexity = item.get("k_hop", "")
                
                # Extract the reasoning length (num2) from "task_[num1].[num2]"
                try:
                    reasoning_length = int(task_complexity)
                except (IndexError, ValueError):
                    reasoning_length = 0 # Fallback if data is malformed
                
                prompt = prompt_function(story, query, examples)
                
                batch.append({
                    "question": prompt,
                    "answer": target,
                    "task_type": "stepgame",
                    "metadata": {
                        "story": story,
                        "query": query,
                        "task_name": task_complexity,
                        "reasoning_length": reasoning_length,
                        "split_origin": split_name
                    }
                })
                
        print(f"Loaded a total of {len(batch)} problems across all splits.")
        return batch

    @staticmethod
    def build_prompt_stepgame_baseline(story: str, query: str) -> str:
        name1, name2 = StepGameManager.extract_query_names(query)

        # 1. Place the strict constraint at the absolute bottom (Recency Bias)
        prompt = f"""Story:
{story}

Possible answers:
- above
- below
- left
- lower-left
- lower-right
- overlap
- right
- upper-left
- upper-right

Task: State only the one kinship word (from the posible answers) that describes the spatial relationship between {name2} and {name1}. {query}"""

        baseline_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        
        return baseline_prompt
    
    @staticmethod
    def build_prompt_clutrr_few_shots_try_examples(story: str, query: str, examples: str) -> str:
        name1, name2 = StepGameManager.extract_query_names(query)

        # 1. Compress options to save context window and improve attention gravity
        options = "above, below, left, lower-left, lower-right, overlap, right, upper-left, upper-right"

        # 2. Construct true multi-turn few-shot history        
        few_shots_prompt_0 = f"""<start_of_turn>system
Possible relationships: [{options}]<end_of_turn>
<start_of_turn>user
Story: [Ashley]'s daughter, [Lillian], asked her mom to read her a story. [Nicholas]'s sister [Lillian] asked him for some help planting her garden.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Nicholas is Ashley's?<end_of_turn>
<start_of_turn>model
son<end_of_turn>
<start_of_turn>user
Story: [June] went with her husband [James] to get a nice dinner for their anniversary. [Dale] is taking his son [James] out for coffee.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. June is Dale's?<end_of_turn>
<start_of_turn>model
daughter-in-law<end_of_turn>
<start_of_turn>user
Story: [Wayne] was looking forward to his wife [Nancy] coming back home. She was away for the weekend with her daughter [Lorraine].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Lorraine is Wayne's?<end_of_turn>
<start_of_turn>model
daughter<end_of_turn>
<start_of_turn>user
Story: {story}
CRITICAL TASK: State the spatial relationship. Output EXACTLY ONE WORD from the list above. {query}<end_of_turn>
<start_of_turn>model
"""        
        
        final_part = f"""<start_of_turn>user
Story: {story}
CRITICAL TASK: State the spatial relationship. Output EXACTLY ONE WORD from the list above. {query}<end_of_turn>
<start_of_turn>model
"""        
        few_shots_prompt = examples + final_part
        
        return few_shots_prompt
    
    @staticmethod
    def build_prompt_clutrr_few_shots(story: str, query: str) -> str:
        name1, name2 = StepGameManager.extract_query_names(query)

        # 1. Compress options to save context window and improve attention gravity
        options = "above, below, left, lower-left, lower-right, overlap, right, upper-left, upper-right"
        # 2. Construct true multi-turn few-shot history        
        few_shots_prompt = f"""<start_of_turn>system
Possible relationships: [{options}]<end_of_turn>
<start_of_turn>user
Story: [Seth] and his grandmother [Mary] went to the science museum. They both had fun, and learned some things, too. [Arthur] enjoys going fishing with his brother. His name is [Warren]. [Seth] and his brother [Warren] always played pranks on each other [Warren] was disappointed that his father, [Alvin], would n't be at the play to see him perform. [Warren] took his son [Alvin] out to play gold later that night. [Warren] played chess with his brother [Arthur].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Mary is Warren's?<end_of_turn>
<start_of_turn>model
grandmother<end_of_turn>
<start_of_turn>user
Story: [Ross] went to his brother [Michael]'s Birthday party [Ronald]'s aunt [Erica] likes to drink a little bit too much wine at family gatherings. [Patrick] asked his brother [Ross] if he would come help him fix his car next weekend. [Erica] was mad at her son, [Michael]. She found he'd been stealing from her purse. [Robert] would n't let his son [James] go to the park by himself. [James]'s brother [Ronald] offered to go with him.
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Patrick is Robert's?<end_of_turn>
<start_of_turn>model
nephew<end_of_turn>
<start_of_turn>user
Story: [Patrick]'s father, [Joseph], went bowling with his sister, [Katherine]. [Katherine] and her son [Ronald] went out to lunch together yesterday. [Alfredo] went to the Farmer's market with his mother [Erica] and his brother [Patrick]. [Ronald] went to the game with his sister [Charlsie].
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. Charlsie is Erica's?<end_of_turn>
<start_of_turn>model
niece<end_of_turn>
<start_of_turn>user
Story: {story}
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {name2} is {name1}'s?<end_of_turn>
<start_of_turn>model
"""
        
        return few_shots_prompt
    
    @staticmethod
    def build_prompt_clutrr(story: str, query: str) -> str:
        name1, name2 = StepGameManager.extract_query_names(query)

        # OPTIMIZATION: Naming the entities directly in the task line
        # to maximize attention gravity right before generation.
        prompt = f"""Story:
{story}

Possible answers:
- above
- below
- left
- lower-left
- lower-right
- overlap
- right
- upper-left
- upper-right

Understand the spatial relationship between {name2} and {name1}, and to describe it through only one kinship word from the posible answers, to correctly answer the question: {query}"""
        
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
    
    
##---------- FUNCTIONS FOR CURATED DATASET CREATION AND FREEZING ----------##

    def save_dataset_to_json(self, dataset: list[dict], filepath: str = "Data/curated_clutrr_subset.json"):
        """
        Serializes the generated dataset to a JSON file to guarantee repeatability.
        Uses indent=4 to keep the file human-readable for debugging.
        """
        # Ensure the target directory exists
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        try:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=4, ensure_ascii=False)
            print(f"\n[IO] Dataset successfully permanently saved to: {os.path.abspath(filepath)}")
        except Exception as e:
            print(f"\n[IO Error] Failed to save dataset: {e}")

    def load_dataset_from_json(self, filepath: str = "Data/curated_clutrr_subset.json") -> list[dict]:
        """
        Loads a strictly frozen dataset from a JSON file.
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                dataset = json.load(f)
            print(f"\n[IO] Frozen dataset loaded successfully from: {os.path.abspath(filepath)}")
            return dataset
        except FileNotFoundError:
            print(f"\n[IO Error] File not found: {filepath}")
            return []
        except json.JSONDecodeError:
            print(f"\n[IO Error] File {filepath} is corrupted or not valid JSON.")
            return []
    
    def get_curated_dataset(self) -> list[dict]:
        # 1. Group actual data by Relation -> Complexity -> List of Items
        data_store = defaultdict(lambda: defaultdict(list))
        by_complexity = defaultdict(list)

        for split_name in self.dataset.keys():
            for item in self.dataset[split_name]:
                target = item.get("label", "")
                task_name = item.get("k_hop", "")
                
                if not target:
                    continue
                reasoning_length = int(task_name)

                story = item.get("story", "")
                query = item.get("query", "")

                prompt = self.build_prompt_clutrr(story, query)
                
                # Store the formatted item
                formatted_item = {
                    "question": prompt,
                    "answer": target,
                    "task_type": "stepgame",
                    "metadata": {
                        "story": story,
                        "query": query,
                        "task_name": task_name,
                        "reasoning_length": reasoning_length,
                        "injected_noise": "",
                        "split_origin": split_name
                    },
                    "id": item.get("id", None)
                }     

                data_store[target][reasoning_length].append(formatted_item)
                by_complexity[reasoning_length].append(formatted_item)
                # --- n=0: Original Behavior ---
        if n == 0:
            curated_dataset = []
            for relation, complexities_dict in data_store.items():
                complexities = list(complexities_dict.keys())
                random.shuffle(complexities)
                for i in range(5):
                    for attempts in range(len(complexities)):
                        target_k = complexities[(i + attempts) % len(complexities)]
                        bucket = complexities_dict[target_k]
                        if bucket:
                            curated_dataset.append(bucket.pop(random.randrange(len(bucket))))
                            break
            random.shuffle(curated_dataset)
            return curated_dataset

        # --- n=1: Stratified Sampling ---
        elif n == 1:
            curated_dataset = []
            # Take 28 from levels 2, 3, 4
            for level in [2, 3, 4]:
                curated_dataset.extend(random.sample(by_complexity[level], 28))
            # Take 1 from all other levels (2, 3, 4 are already done)
            for level in by_complexity.keys():
                if level not in [2, 3, 4]:
                    curated_dataset.extend(random.sample(by_complexity[level], 1))
            return curated_dataset

        # --- n=2: Specific Level Sampling ---
        elif n == 2:
            # Take 90 from level 6
            return random.sample(by_complexity[6], 90)
        
        return []

    def get_or_create_curated_dataset(self):

        SAVE_PATH = "Data/curated_clutrr_subset.json"
        
        # 1. Check if we already have a frozen dataset
        if os.path.exists(SAVE_PATH):
            final_data = self.load_dataset_from_json(SAVE_PATH)
            
            # Quick validation of the loaded data
            print(f"Loaded {len(final_data)} evaluation elements ready for inference.")
            
        # 2. If no frozen data exists, generate it from scratch and lock it
        else:
            final_data = self.get_curated_dataset()
            
            if final_data:
                self.save_dataset_to_json(final_data, SAVE_PATH)
        
        return final_data

