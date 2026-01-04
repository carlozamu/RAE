def create_id(class_type: type) -> int:
    pass

NEXT_INNOVATION_NUMBER = -1
def _get_next_innovation_number() -> int:
    global NEXT_INNOVATION_NUMBER
    NEXT_INNOVATION_NUMBER += 1
    return NEXT_INNOVATION_NUMBER

import matplotlib.pyplot as plt
import numpy as np
import os
import random
from typing import Tuple, List
from Phenotype.phenotype import Phenotype

def plot_complexity_vs_fitness(generation_data: list[Tuple[int, Phenotype]], generation_idx: int, species_colors_registry: dict[str, str], output_dir="plots") -> str:
    """
    Plots Complexity vs Fitness using a CUSTOM EQUIDISTANT SCALE.
    Uses strict string casting to ensure consistent colors across generations.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # --- 1. Define the Custom Grid ---
    x_anchors = [0.1, 0.3, 0.5, 0.8, 1.5, 2.5]
    y_anchors = [1, 2, 5, 10, 20, 50, 100]

    x_positions = np.arange(len(x_anchors))
    y_positions = np.arange(len(y_anchors))

    # --- 2. Process Data & Assign Colors ---
    mapped_xs = []
    mapped_ys = []
    colors = []
    
    for species_id, phenotype in generation_data:
        # A. Transform Coordinates
        real_loss = phenotype.genome.fitness
        num_nodes = len(phenotype.genome.nodes)
        num_conns = len([c for c in phenotype.genome.connections.values() if c.enabled])
        real_complexity = max(1, num_nodes + num_conns)
        
        m_x = np.interp(real_loss, x_anchors, x_positions)
        m_y = np.interp(real_complexity, y_anchors, y_positions)
        
        mapped_xs.append(m_x)
        mapped_ys.append(m_y)
        
        # B. Robust Color Management (STRICT CASTING)
        # We cast the ID to string immediately. This is the master key.
        lookup_key = str(species_id)
        
        if lookup_key not in species_colors_registry:
            # Generate new color only if this specific string key is missing
            r = random.randint(30, 200)
            g = random.randint(30, 200)
            b = random.randint(30, 200)
            species_colors_registry[lookup_key] = f'#{r:02x}{g:02x}{b:02x}'
            
        colors.append(species_colors_registry[lookup_key])

    # --- 3. Create Plot ---
    plt.figure(figsize=(10, 7))
    
    plt.scatter(mapped_xs, mapped_ys, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    # --- 4. Fake the Axis Labels ---
    plt.xticks(x_positions, [str(x) for x in x_anchors])
    plt.yticks(y_positions, [str(y) for y in y_anchors])
    
    plt.xlim(-0.5, len(x_anchors) - 0.5)
    plt.ylim(-0.5, len(y_anchors) - 0.5)

    plt.title(f"Generation {generation_idx}: Complexity vs. Loss")
    plt.xlabel("Loss (Fitness) -> Lower is Better")
    plt.ylabel("Complexity (Nodes + Enabled Edges)")
    plt.grid(True, linestyle='--', alpha=0.5)

    # Legend Logic
    # We use the same casting logic to group items for the legend
    active_species_ids = set(str(item[0]) for item in generation_data)
    legend_handles = []
    
    # Sort them so the legend order doesn't jump around randomly
    for s_key in sorted(list(active_species_ids)):
        color = species_colors_registry[s_key]
        patch = plt.Line2D([0], [0], marker='o', color='w', label=f"Spec {s_key[:6]}", 
                          markerfacecolor=color, markersize=10)
        legend_handles.append(patch)
    
    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    # 5. Save
    filename = f"{output_dir}/gen_{generation_idx}_scatter.png"
    plt.savefig(filename)
    plt.close()
    
    return filename
