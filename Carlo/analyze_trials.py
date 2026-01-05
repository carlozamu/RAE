
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter

def analyze_history(file_path):
    generations = []
    max_fitnesses = []
    avg_fitnesses = []
    max_complexities = []
    avg_complexities = []
    species_counts = []
    
    best_genomes_over_time = []

    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
                
            gen = data.get('generation')
            pop = data.get('population', [])
            
            if not pop: continue
            
            fitnesses = [ind.get('fitness', 0) for ind in pop]
            complexities = [ind.get('num_nodes', 0) + ind.get('num_connections_enabled', 0) for ind in pop]
            species = set(ind.get('species_id') for ind in pop)
            
            generations.append(gen)
            max_fitnesses.append(max(fitnesses))
            avg_fitnesses.append(sum(fitnesses) / len(fitnesses))
            
            max_complexities.append(max(complexities))
            avg_complexities.append(sum(complexities) / len(complexities))
            
            species_counts.append(len(species))
            
            # Extract details of the best individual
            best_ind =  max(pop, key=lambda x: x.get('fitness', 0))
            best_dump = data.get('best_individual_dump')
            if best_dump:
                 best_genomes_over_time.append(best_dump)
            else:
                # Fallback if dump not present, though usually is at end of json
                pass

    print(f"Parsed {len(generations)} generations.")
    print(f"Final Generation: {generations[-1]}")
    print(f"Final Max Fitness: {max_fitnesses[-1]}")
    print(f"Final Avg Fitness: {avg_fitnesses[-1]}")
    print(f"Final Max Complexity: {max_complexities[-1]}")
    
    # Analyze trends
    correlation = np.corrcoef(max_fitnesses, max_complexities)[0, 1]
    print(f"Correlation between Max Fitness and Max Complexity: {correlation:.4f}")

    # Analyze Best Genome Instructions
    if best_genomes_over_time:
        last_best = best_genomes_over_time[-1]
        print("\n=== BEST INDIVIDUAL (Latest) ===")
        print(f"ID: {last_best.get('genome_id')}")
        print(f"Fitness: {last_best.get('final_fitness')}")
        print("Instructions:")
        nodes = sorted(last_best.get('nodes', []), key=lambda x: x.get('innovation_number'))
        
        # Try to sort topologically if connections present, otherwise just ID
        # Simple print for now
        for n in nodes:
            print(f"  - [{n.get('name')}]: {n.get('instruction')}")
            
    # Check for Split Mutation evidence (checking if any instruction looks like a split of another or specific keywords)
    # We can look at the "instruction" content for words like "Step 1", "First", etc.
    
    print("\n=== PATTERN ANALYSIS ===")
    instruction_keywords = []
    for bg in best_genomes_over_time[-10:]: # Look at last 10 best
        for n in bg.get('nodes', []):
            instruction_keywords.extend(n.get('instruction', '').split())
    
    common_words = Counter([w.lower() for w in instruction_keywords if len(w) > 3]).most_common(10)
    print("Most common words in best instructions (last 10 gens):", common_words)

if __name__ == "__main__":
    analyze_history("analysis_data.jsonl")
