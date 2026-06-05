import re
from collections import defaultdict
from datasets import load_dataset

def profile_dataset_with_checksums():
    split_config = "gen_train234_test2to10"
    
    print(f"Loading CLUTTR dataset: CLUTRR/v1 - {split_config}...")
    try:
        # Using trust_remote_code=True for datasets v2.19.0 compatibility
        dataset = load_dataset("CLUTRR/v1", split_config, trust_remote_code=True)
        print("CLUTTR dataset loaded successfully.\n")
    except Exception as e:
        print(f"Error loading CLUTTR dataset: {e}")
        return

    # 1. Calculate Absolute Total Upfront
    absolute_total_elements = sum(len(dataset[split]) for split in dataset.keys())
    print(f"INIT: Total elements reported by dataset object across all splits: {absolute_total_elements}\n")

    distribution = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    unique_relations = set()
    
    # Tracking variables for our checksum
    processed_count = 0
    skipped_no_target = 0
    skipped_malformed_task = 0
    
    for split_name in dataset.keys():
        data_split = dataset[split_name]
        
        for item in data_split:
            target = item.get("target_text", "")
            task_name = item.get("task_name", "")
            
            # Safeguard 1: Missing Target
            if not target:
                skipped_no_target += 1
                continue
            
            # Safeguard 2: Regex extraction
            match = re.search(r"task_(\d+)\.(\d+)", task_name)
            if match:
                injected_noise = int(match.group(1))
                reasoning_length = int(match.group(2))
            else:
                skipped_malformed_task += 1
                continue 
            
            # If it passes both safeguards, log it
            unique_relations.add(target)
            distribution[target][reasoning_length][injected_noise] += 1
            processed_count += 1

    # --- PRINTING DISTRIBUTION ---
    print("=== DATASET DISTRIBUTION PROFILE ===")
    print(f"Total unique relations discovered: {len(unique_relations)}\n")
    
    for relation in sorted(list(unique_relations)):
        print(f"Target Relation: [{relation.upper()}]")
        for complexity in sorted(distribution[relation].keys()):
            noise_counts = distribution[relation][complexity]
            total_for_complexity = sum(noise_counts.values())
            noise_breakdown = ", ".join([f"Noise {n}: {count}" for n, count in sorted(noise_counts.items())])
            print(f"  Complexity k={complexity} (Total: {total_for_complexity}) -> {noise_breakdown}")
        print("") 

    # --- PRINTING CHECKSUMS ---
    print("=== PIPELINE CHECKSUM DIAGNOSTICS ===")
    print(f"Absolute Initial Total: {absolute_total_elements}")
    print(f"Total Successfully Processed: {processed_count}")
    print(f"Total Skipped (No Target): {skipped_no_target}")
    print(f"Total Skipped (Malformed Task Name): {skipped_malformed_task}")
    
    calculated_total = processed_count + skipped_no_target + skipped_malformed_task
    
    print("-" * 40)
    if absolute_total_elements == calculated_total:
        print(f"VERIFICATION PASS: {absolute_total_elements} == {calculated_total}. No data was lost in the loop.")
    else:
        print(f"VERIFICATION FAIL: Expected {absolute_total_elements}, but got {calculated_total}. Check loop logic.")

    # --- PRINTING CLASS DISTRIBUTION BY COMPLEXITY ---
    print("\n=== CLASS DISTRIBUTION BY COMPLEXITY ===")
    
    # Invert the dictionary: complexity -> target -> total_count
    complexity_distribution = defaultdict(dict)
    
    for relation, complexities in distribution.items():
        for complexity, noises in complexities.items():
            # Sum across noises (even though we know it's only Noise 1 here)
            total = sum(noises.values())
            if total > 0:
                complexity_distribution[complexity][relation] = total

    # Print grouped by complexity
    for complexity in sorted(complexity_distribution.keys()):
        print(f"Complexity k={complexity}")
        
        # Sort relations alphabetically for cleaner output
        relation_counts = complexity_distribution[complexity]
        for relation in sorted(relation_counts.keys()):
            print(f"  -> [{relation.upper()}]: {relation_counts[relation]}")
        
        # Show total number of available classes at this complexity level
        print(f"  [Total unique classes available at k={complexity}: {len(relation_counts)}]\n")
        
if __name__ == "__main__":
    profile_dataset_with_checksums()