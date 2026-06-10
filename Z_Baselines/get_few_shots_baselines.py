import random
from collections import defaultdict

from Data.clutrr import CLUTTRManager

def extract_names(query: str) -> tuple[str, str]:
    """Helper to extract name1 and name2 from the CLUTRR query string."""
    clean_query = query.replace("(", "").replace(")", "").replace("'", "")
    try:
        name1, name2 = [name.strip() for name in clean_query.split(',')]
    except ValueError:
        name1, name2 = "Person A", "Person B"
    return name1, name2

def generate_prompt_templates(dataset_batch: list[dict], output_filename: str = "generated_prompts.py"):
    """
    Groups the stratified dataset by reasoning length, samples examples for the 10 configurations,
    and generates fully formatted few-shot prompt templates.
    """
    # 1. Configuration Definitions
    configurations = {
        "min_complexity": [2, 2, 2],
        "max_extrapolation": [10, 10, 10],
        "uniform_median": [6, 6, 6],
        "bounding_box": [2, 6, 10],
        "inverted_recency": [10, 6, 2],
        "strided_generalist": [2, 5, 8],
        "dist_aligned_control": [2, 3, 4],
        "boundary_pusher": [3, 4, 5],
        "hard_tail_anchor": [8, 9, 10],
        "bimodal_catalyst": [2, 2, 10]        
    }

    # 2. Data Grouping
    # Group all available examples by their reasoning_length (hops)
    examples_by_length = defaultdict(list)
    for item in dataset_batch:
        length = item["metadata"]["reasoning_length"]
        # Only store the necessary fields for building the prompt
        examples_by_length[length].append({
            "story": item["metadata"]["story"],
            "query": item["metadata"]["query"],
            "answer": item["answer"]
        })
    
    generated_templates = {}
    options_string = "aunt, son-in-law, grandfather, brother, sister, father, mother, grandmother, uncle, daughter-in-law, grandson, granddaughter, father-in-law, mother-in-law, nephew, son, daughter, niece"

    # 3. Controlled Sampling and Template Generation
    for config_name, hops in configurations.items():
        few_shot_blocks = []
        
        # Sample exactly one example for each hop required in the current combination
        for hop in hops:
            if not examples_by_length[hop]:
                raise ValueError(f"No examples found in the dataset for reasoning length: {hop}")
            
            # Randomly select an example for this specific hop
            sampled_example = random.choice(examples_by_length[hop])
            name1, name2 = extract_names(sampled_example["query"])
            
            # Construct the chat block for this specific example
            block = (
                f"<start_of_turn>user\n"
                f"Story: {sampled_example['story']}\n"
                f"CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {name2} is {name1}'s?<end_of_turn>\n"
                f"<start_of_turn>model\n"
                f"{sampled_example['answer']}<end_of_turn>"
            )
            few_shot_blocks.append(block)
        
        # Join the sampled blocks together
        history_string = "\n".join(few_shot_blocks)
        
        # Build the final f-string template, leaving {story}, {name1}, and {name2} unformatted
        # using double braces {{ }} to escape them in the output code.
        full_template = (
            f"f\"\"\"<start_of_turn>system\n"
            f"Possible relationships: [{options_string}]<end_of_turn>\n"
            f"{history_string}\n"
            f"<start_of_turn>user\n"
            f"Story: {{story}}\n"
            f"CRITICAL TASK: State the family relationship. Output EXACTLY ONE WORD from the list above. {{name2}} is {{name1}}'s?<end_of_turn>\n"
            f"<start_of_turn>model\n"
            f"\"\"\""
        )
        
        generated_templates[config_name] = full_template

    # 4. Serialization
    # Write the templates into a valid Python file as a dictionary
    with open(output_filename, "w", encoding="utf-8") as f:
        f.write("# AUTO-GENERATED PROMPT TEMPLATES\n\n")
        f.write("PROMPT_TEMPLATES = {\n")
        for config_name, template_str in generated_templates.items():
            f.write(f"    '{config_name}': {template_str},\n")
        f.write("}\n")

    print(f"Successfully generated {len(configurations)} prompt templates and saved to {output_filename}.")

# ==========================================
# Execution Hook
# ==========================================
if __name__ == "__main__":
    clutrr_manager = CLUTTRManager()
    raw_batch = clutrr_manager.get_entire_dataset_stratified(clutrr_manager.build_prompt_clutrr_few_shots)
    
    generate_prompt_templates(raw_batch)