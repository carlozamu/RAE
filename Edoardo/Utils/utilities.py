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
from Phenotype.phenotype import Phenotype  # Ensure this import matches your project structure

def plot_complexity_vs_fitness(generation_data: list[Tuple[int, Phenotype]], generation_idx: int, species_colors_registry: dict[int, str], output_dir="plots") -> str:
    """
    Plots Complexity vs Fitness with LOGARITHMIC SCALES.
    X-Axis: Loss (SymLog to handle 0). Ticks: [0, 0.3, 0.6, 1, 1.5, 2, 3]
    Y-Axis: Complexity (Log). Ticks: [1, 2, 5, 10, 20, 50, 100]
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
        # Ensure Y is at least 1 to avoid log(0) errors if an agent is somehow empty
        ys.append(max(1, num_nodes + num_conns))
        
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
    
    # 4. LOGARITHMIC AXIS CONFIGURATION
    # ---------------------------------------------------------
    
    # --- Y-Axis: Standard Log Scale ---
    plt.yscale('log')
    plt.ylim(0.9, 110) # Set limits slightly wider than [1, 100] so dots aren't cut off
    
    # Custom Ticks for Y
    y_ticks = [1, 2, 5, 10, 20, 50, 100]
    plt.yticks(y_ticks, [str(y) for y in y_ticks])

    # --- X-Axis: Symmetrical Log Scale (SymLog) ---
    # SymLog behaves like log, but is linear around zero (linthresh determines the linear range)
    # This allows us to plot 0 without error.
    plt.xscale('symlog', linthresh=0.1) 
    plt.xlim(-0.05, 3.5) # Start slightly below 0 to visualize the 0 line clearly
    
    # Custom Ticks for X as requested
    x_ticks = [0, 0.3, 0.6, 1, 1.5, 2, 3]
    plt.xticks(x_ticks, [str(x) for x in x_ticks])
    
    # ---------------------------------------------------------

    plt.title(f"Generation {generation_idx}: Complexity vs. Loss (Log Scales)")
    plt.xlabel("Loss (Fitness) -> Lower is Better")
    plt.ylabel("Complexity (Nodes + Enabled Edges)")
    
    # Add grid (using 'both' ensures minor log grid lines appear too)
    plt.grid(True, which="major", linestyle='-', alpha=0.6)
    plt.grid(True, which="minor", linestyle=':', alpha=0.3)
    
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
