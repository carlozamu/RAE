from collections import defaultdict
from datasets import load_dataset


def profile_dataset_with_checksums():
    print("Loading StepGame dataset...")
    try:
        dataset = load_dataset("ZhengyanShi/StepGame", trust_remote_code=True)
        print("StepGame dataset loaded successfully.\n")
    except Exception as e:
        print(f"Error loading StepGame dataset: {e}")
        return

    # 1. Calculate absolute total upfront.
    absolute_total_elements = sum(len(dataset[split]) for split in dataset.keys())
    print(f"INIT: Total elements reported by dataset object across all splits: {absolute_total_elements}\n")

    # distribution[label][k_hop] -> count
    distribution = defaultdict(lambda: defaultdict(int))
    unique_labels = set()

    # Tracking variables for checksum diagnostics.
    processed_count = 0
    skipped_no_label = 0
    skipped_invalid_k_hop = 0

    for split_name in dataset.keys():
        data_split = dataset[split_name]

        for item in data_split:
            label = item.get("label", "")
            k_hop = item.get("k_hop", None)

            if not label:
                skipped_no_label += 1
                continue

            # k_hop should be numeric and castable to int for grouping.
            try:
                complexity = int(k_hop)
            except (TypeError, ValueError):
                skipped_invalid_k_hop += 1
                continue 

            unique_labels.add(label)
            distribution[label][complexity] += 1
            processed_count += 1

    # --- PRINTING DISTRIBUTION ---
    print("=== DATASET DISTRIBUTION PROFILE ===")
    print(f"Total unique labels discovered: {len(unique_labels)}\n")

    for label in sorted(unique_labels):
        print(f"Target Label: [{label.upper()}]")
        for complexity in sorted(distribution[label].keys()):
            total_for_complexity = distribution[label][complexity]
            print(f"  Complexity k={complexity} (Total: {total_for_complexity})")
        print("") 

    # --- PRINTING CHECKSUMS ---
    print("=== PIPELINE CHECKSUM DIAGNOSTICS ===")
    print(f"Absolute Initial Total: {absolute_total_elements}")
    print(f"Total Successfully Processed: {processed_count}")
    print(f"Total Skipped (No Label): {skipped_no_label}")
    print(f"Total Skipped (Invalid k_hop): {skipped_invalid_k_hop}")

    calculated_total = processed_count + skipped_no_label + skipped_invalid_k_hop

    print("-" * 40)
    if absolute_total_elements == calculated_total:
        print(f"VERIFICATION PASS: {absolute_total_elements} == {calculated_total}. No data was lost in the loop.")
    else:
        print(f"VERIFICATION FAIL: Expected {absolute_total_elements}, but got {calculated_total}. Check loop logic.")

    # --- PRINTING CLASS DISTRIBUTION BY COMPLEXITY ---
    print("\n=== CLASS DISTRIBUTION BY COMPLEXITY ===")

    # Invert dictionary to: complexity -> label -> total_count
    complexity_distribution = defaultdict(dict)

    for label, complexities in distribution.items():
        for complexity, count in complexities.items():
            complexity_distribution[complexity][label] = count

    # Print grouped by complexity
    for complexity in sorted(complexity_distribution.keys()):
        print(f"Complexity k={complexity}")

        # Sort labels alphabetically for cleaner output.
        label_counts = complexity_distribution[complexity]
        for label in sorted(label_counts.keys()):
            print(f"  -> [{label.upper()}]: {label_counts[label]}")

        # Show total number of available classes at this complexity level
        print(f"  [Total unique classes available at k={complexity}: {len(label_counts)}]\n")

if __name__ == "__main__":
    profile_dataset_with_checksums()