import json
import re
import random
from typing import List, Dict, Tuple, Any, Optional
from datasets import load_dataset
import os
from collections import defaultdict

class CLUTTRManager:
    RELATIONS = [
        "aunt", "son-in-law", "grandfather", "brother", "sister",
        "father", "mother", "grandmother", "uncle", "daughter-in-law",
        "grandson", "granddaughter", "father-in-law", "mother-in-law",
        "nephew", "son", "daughter", "niece"
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
                query = item["query"]
                target = item["target_text"]
                task_name = item.get("task_name", "")
                
                # Extract the reasoning length (num2) from "task_[num1].[num2]"
                try:
                    reasoning_length = int(task_name.split(".")[1])
                except (IndexError, ValueError):
                    reasoning_length = 0 # Fallback if data is malformed
                
                prompt = prompt_function(story, query, examples)
                
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

Task: State only the one kinship word (from the posible answers) that describes the family relationship between {name2} and {name1}. {name2} is {name1}'s?"""

        baseline_prompt = f"<start_of_turn>user\n{prompt}\n<end_of_turn>\n<start_of_turn>model\n"
        
        return baseline_prompt
    
    @staticmethod
    def build_prompt_clutrr_few_shots_try_examples(story: str, query: str, examples: str) -> str:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # 1. Compress options to save context window and improve attention gravity
        options = "aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece"

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
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {name2} is {name1}'s?<end_of_turn>
<start_of_turn>model
"""        
        
        final_part = f"""<start_of_turn>user
Story: {story}
CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {name2} is {name1}'s?<end_of_turn>
<start_of_turn>model
"""        
        few_shots_prompt = examples + final_part
        
        return few_shots_prompt
    
    @staticmethod
    def build_prompt_clutrr_few_shots(story: str, query: str) -> str:
        clean_query = query.replace("(", "").replace(")", "").replace("'", "")
        try:
            name1, name2 = [name.strip() for name in clean_query.split(',')]
        except ValueError:
            name1, name2 = "Person A", "Person B"

        # 1. Compress options to save context window and improve attention gravity
        options = "aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece"

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
        
        for split_name in self.dataset.keys():
            for item in self.dataset[split_name]:
                target = item.get("target_text", "")
                task_name = item.get("task_name", "")
                
                if not target:
                    continue
                
                match = re.search(r"task_(\d+)\.(\d+)", task_name)
                if match:
                    injected_noise = int(match.group(1))
                    reasoning_length = int(match.group(2))

                    story = item.get("story", "")
                    query = item.get("query", "")

                    prompt = self.build_prompt_clutrr(story, query)
                    
                    # Store the formatted item
                    data_store[target][reasoning_length].append({
                        "question": prompt,
                        "answer": target,
                        "task_type": "cluttr",
                        "metadata": {
                            "story": story,
                            "query": query,
                            "task_name": task_name,
                            "reasoning_length": reasoning_length,
                            "injected_noise": injected_noise,
                            "split_origin": split_name
                        },
                        "id": item.get("id", None)
                    })             
        
        # 2. Sample 5 elements per relation using the modulo/shuffle algorithm
        curated_dataset = []
        
        for relation, complexities_dict in data_store.items():
            complexities = list(complexities_dict.keys())
            random.shuffle(complexities) 
            
            num_complexities = len(complexities)
            sampled_for_relation = 0
            
            for i in range(5):
                attempts = 0
                item_found = False
                
                while attempts < num_complexities:
                    target_idx = (i + attempts) % num_complexities
                    target_k = complexities[target_idx]
                    
                    bucket = complexities_dict[target_k]
                    
                    if len(bucket) > 0:
                        random_idx = random.randrange(len(bucket))
                        selected_item = bucket.pop(random_idx)
                        
                        curated_dataset.append(selected_item)
                        sampled_for_relation += 1
                        item_found = True
                        break
                    
                    attempts += 1
                
                if not item_found:
                    print(f"Warning: Dataset exhausted for [{relation.upper()}] after {sampled_for_relation} samples.")
                    break 
                    
        print(f"Extraction complete. Total curated elements: {len(curated_dataset)}\n")
        
        # Optional: Shuffle the final dataset
        random.shuffle(curated_dataset)

        # ==========================================
        # 3. VERIFICATION AND ANALYSIS PRINTS
        # ==========================================
        
        # Rebuild distribution dictionary specifically from the extracted subset
        # curated_distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        
        # for item in curated_dataset:
        #     target = item["answer"]
        #     complexity = item["metadata"]["reasoning_length"]
        #     noise = item["metadata"]["injected_noise"]
        #     curated_distribution[target][complexity][noise] += 1

        # # Print 1: Dataset Distribution Profile
        # print("=== CURATED DATASET DISTRIBUTION PROFILE ===")
        # for relation in sorted(curated_distribution.keys()):
        #     print(f"Target Relation: [{relation.upper()}]")
        #     total_for_relation = 0
            
        #     for complexity in sorted(curated_distribution[relation].keys()):
        #         noise_counts = curated_distribution[relation][complexity]
        #         total_for_complexity = sum(noise_counts.values())
        #         total_for_relation += total_for_complexity
                
        #         noise_breakdown = ", ".join([f"Noise {n}: {count}" for n, count in sorted(noise_counts.items())])
        #         print(f"  Complexity k={complexity} (Total: {total_for_complexity}) -> {noise_breakdown}")
                
        #     print(f"  [Total enforced for {relation.upper()}: {total_for_relation}]\n")

        # # Print 2: Class Distribution by Complexity
        # print("=== CURATED CLASS DISTRIBUTION BY COMPLEXITY ===")
        # complexity_distribution = defaultdict(dict)
        
        # for relation, complexities in curated_distribution.items():
        #     for complexity, noises in complexities.items():
        #         total = sum(noises.values())
        #         if total > 0:
        #             complexity_distribution[complexity][relation] = total

        # for complexity in sorted(complexity_distribution.keys()):
        #     print(f"Complexity k={complexity}")
            
        #     relation_counts = complexity_distribution[complexity]
        #     for relation in sorted(relation_counts.keys()):
        #         print(f"  -> [{relation.upper()}]: {relation_counts[relation]}")
                
        #     print(f"  [Total classes represented at k={complexity}: {len(relation_counts)}]\n")

        return curated_dataset

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

