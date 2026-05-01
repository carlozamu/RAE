import os
import random
import json
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from Phenotype.phenotype import Phenotype

# --- 1. ID Generators ---
_ID_REGISTRY = {}

def create_id(class_type: type) -> int:
    """Generates a unique incremental ID per class type."""
    type_name = class_type.__name__
    if type_name not in _ID_REGISTRY:
        _ID_REGISTRY[type_name] = -1
    _ID_REGISTRY[type_name] += 1
    return _ID_REGISTRY[type_name]

NEXT_INNOVATION_NUMBER = -1
def _get_next_innovation_number() -> int:
    global NEXT_INNOVATION_NUMBER
    NEXT_INNOVATION_NUMBER += 1
    return NEXT_INNOVATION_NUMBER


# --- 2. Plotting Utility ---
def plot_complexity_vs_fitness(generation_data: List[Phenotype], generation_idx: int, species_colors_registry: dict[str, str], output_dir="plots") -> str:
    """
    Plots Complexity vs Fitness using a CUSTOM EQUIDISTANT SCALE.
    Updated to handle flat Phenotype lists and a 0-100 Fitness scale.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- A. Define the Custom Grid (Updated for 0-100 Fitness) ---
    x_anchors = [0, 20, 40, 60, 80, 100]  # Spanning the new Fitness scale
    y_anchors = [1, 2, 5, 10, 20, 50, 100]

    x_positions = np.arange(len(x_anchors))
    y_positions = np.arange(len(y_anchors))

    mapped_xs = []
    mapped_ys = []
    colors = []
    
    # --- B. Process Data & Assign Colors ---
    for phenotype in generation_data:
        # Extract metrics
        real_fitness = phenotype.genome.fitness
        num_nodes = len(phenotype.genome.nodes)
        num_conns = len([c for c in phenotype.genome.connections.values() if c.enabled])
        real_complexity = max(1, num_nodes + num_conns)
        
        # Safe extraction of species ID (defaults to "Unknown" if not assigned by macro-layer)
        species_id = getattr(phenotype.genome, 'species_id', "Unknown")
        
        m_x = np.interp(real_fitness, x_anchors, x_positions)
        m_y = np.interp(real_complexity, y_anchors, y_positions)
        
        mapped_xs.append(m_x)
        mapped_ys.append(m_y)
        
        # Robust Color Management
        lookup_key = str(species_id)
        if lookup_key not in species_colors_registry:
            r, g, b = random.randint(30, 200), random.randint(30, 200), random.randint(30, 200)
            species_colors_registry[lookup_key] = f'#{r:02x}{g:02x}{b:02x}'
            
        colors.append(species_colors_registry[lookup_key])

    # --- C. Create Plot ---
    plt.figure(figsize=(10, 7))
    plt.scatter(mapped_xs, mapped_ys, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    # Fake the Axis Labels
    plt.xticks(x_positions, [str(x) for x in x_anchors])
    plt.yticks(y_positions, [str(y) for y in y_anchors])
    
    plt.xlim(-0.5, len(x_anchors) - 0.5)
    plt.ylim(-0.5, len(y_anchors) - 0.5)

    plt.title(f"Generation {generation_idx}: Complexity vs. Fitness")
    plt.xlabel("Fitness -> Higher is Better")
    plt.ylabel("Complexity (Nodes + Enabled Edges)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # --- D. Legend Logic ---
    active_species_ids = set(str(getattr(p.genome, 'species_id', "Unknown")) for p in generation_data)
    legend_handles = []
    
    for s_key in sorted(list(active_species_ids)):
        color = species_colors_registry[s_key]
        patch = plt.Line2D([0], [0], marker='o', color='w', label=f"Spec {s_key[:6]}", 
                          markerfacecolor=color, markersize=10)
        legend_handles.append(patch)
    
    if legend_handles:
        plt.legend(handles=legend_handles, title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    filename = f"{output_dir}/gen_{generation_idx}_scatter.png"
    plt.savefig(filename)
    plt.close()
    
    return filename


# --- 3. Logging Utility ---
def log_generation_to_json(new_gen: List[Phenotype], generation_idx: int, filename="evolution_history.jsonl"):
    """
    Serializes generation metadata and performs a deep dive log of the 
    best individual (highest fitness) in the population.
    """
    if not new_gen:
        return

    # 1. Find the best individual (MAX fitness)
    best_phenotype = max(new_gen, key=lambda x: x.genome.fitness)
    best_species_id = getattr(best_phenotype.genome, 'species_id', "Unknown")

    # 2. Prepare the high-level log entry
    log_entry = {
        "timestamp": datetime.now().isoformat(),
        "generation": generation_idx,
        "population_size": len(new_gen),
        "best_individual_stats": {
            "species_id": best_species_id,
            "fitness": float(best_phenotype.genome.fitness),
            "complexity": len(best_phenotype.genome.nodes) + len(best_phenotype.genome.connections)
        },
        # THE FULL PHENOTYPE EXPORT
        "best_individual_full": {
            "genome_id": best_phenotype.genome.id,
            "nodes": [
                {
                    "id": n.id,
                    "content": n.instruction  
                } for n in best_phenotype.genome.nodes.values()
            ],
            "connections": [
                {
                    "in": c.in_node, 
                    "out": c.out_node, 
                    "enabled": c.enabled
                } for c in best_phenotype.genome.connections.values()
            ],
            "system_metadata": getattr(best_phenotype, 'metadata', {}) 
        }
    }

    # 3. Append to file
    try:
        with open(filename, "a", encoding="utf-8") as f:
            f.write(json.dumps(log_entry) + "\n")
    except Exception as e:
        print(f"❌ Failed to log full phenotype JSON: {e}")