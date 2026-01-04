def create_id(class_type: type) -> int:
    pass

NEXT_INNOVATION_NUMBER = -1
def _get_next_innovation_number() -> int:
    global NEXT_INNOVATION_NUMBER
    NEXT_INNOVATION_NUMBER += 1
    return NEXT_INNOVATION_NUMBER

from typing import Tuple
from Phenotype.phenotype import Phenotype
import matplotlib.pyplot as plt
import numpy as np # Needed for the interval calculations
import os
import random
from typing import Tuple, List

def plot_complexity_vs_fitness(generation_data: list[Tuple[int, Phenotype]], generation_idx: int, species_colors_registry: dict[int, str], output_dir="plots") -> str:
    """
    Plots Complexity vs Fitness with a FIXED FRAME (0-3.5 Loss, 0-100 Complexity).
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. Prepare Data Containers
    xs = [] # Fitness (Loss)
    ys = [] # Complexity (Nodes + Edges)
    colors = []
    
    # 2. Process Data & Assign Colors
    for species_id, phenotype in generation_data:
        # X-Axis: Fitness (Loss)
        xs.append(phenotype.genome.fitness)
        
        # Y-Axis: Complexity
        num_nodes = len(phenotype.genome.nodes)
        num_conns = len([c for c in phenotype.genome.connections.values() if c.enabled])
        ys.append(num_nodes + num_conns)
        
        # Color Management
        if species_id not in species_colors_registry:
            r = random.randint(30, 200)
            g = random.randint(30, 200)
            b = random.randint(30, 200)
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            species_colors_registry[species_id] = hex_color
            
        colors.append(species_colors_registry[species_id])

    # 3. Create Plot
    plt.figure(figsize=(10, 7))
    
    # Scatter plot
    plt.scatter(xs, ys, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    # 4. FIXED AXIS STYLING
    # ---------------------------------------------------------
    # Set the fixed boundaries
    plt.xlim(0, 3.5)
    plt.ylim(0, 100)

    # Set the fixed intervals (ticks)
    # np.arange(start, stop, step) - stop is exclusive, so we go slightly higher
    plt.xticks(np.arange(0, 3.51, 0.2)) 
    plt.yticks(np.arange(0, 101, 10))

    plt.title(f"Generation {generation_idx}: Complexity vs. Loss")
    plt.xlabel("Loss (Fitness) -> Lower is Better")
    plt.ylabel("Complexity (Nodes + Enabled Edges)")
    
    # Add grid that aligns with our new ticks
    plt.grid(True, linestyle='--', alpha=0.5, which='both')
    # ---------------------------------------------------------
    
    # Legend Logic
    active_species_ids = set(item[0] for item in generation_data)
    legend_handles = []
    for s_id in active_species_ids:
        color = species_colors_registry[s_id]
        label_id = str(s_id)[:6] if isinstance(s_id, str) else str(s_id)
        patch = plt.Line2D([0], [0], marker='o', color='w', label=f"Spec {label_id}", 
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
