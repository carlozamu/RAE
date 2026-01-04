def create_id(class_type: type) -> int:
    pass

NEXT_INNOVATION_NUMBER = -1
def _get_next_innovation_number() -> int:
    global NEXT_INNOVATION_NUMBER
    NEXT_INNOVATION_NUMBER += 1
    return NEXT_INNOVATION_NUMBER

from typing import Tuple
import matplotlib.pyplot as plt
import random
import os

from Phenotype.phenotype import Phenotype

def plot_complexity_vs_fitness(generation_data: list[Tuple[int, Phenotype]], generation_idx: int, species_colors_registry: dict[int, str], output_dir="plots") -> str:
    """
    Plots Complexity vs Fitness for the current generation.
    - generation_data: List of (species_id, Phenotype)
    - Returns: Path to the saved image file
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
        # Count enabled connections for effective complexity
        num_nodes = len(phenotype.genome.nodes)
        num_conns = len([c for c in phenotype.genome.connections.values() if c.enabled])
        ys.append(num_nodes + num_conns)
        
        # Color Management
        if species_id not in species_colors_registry:
            # Generate a random distinctive color for new species
            # Using specific ranges to avoid too light/white colors
            r = random.randint(30, 200)
            g = random.randint(30, 200)
            b = random.randint(30, 200)
            hex_color = f'#{r:02x}{g:02x}{b:02x}'
            species_colors_registry[species_id] = hex_color
            
        colors.append(species_colors_registry[species_id])

    # 3. Create Plot
    plt.figure(figsize=(10, 7))
    
    # Scatter plot with transparency to see overlaps
    plt.scatter(xs, ys, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
    
    # 4. Styling
    plt.title(f"Generation {generation_idx}: Complexity vs. Loss")
    plt.xlabel("Loss (Fitness) -> Lower is Better")
    plt.ylabel("Complexity (Nodes + Enabled Edges)")
    plt.grid(True, linestyle='--', alpha=0.5)
    
    # Add a custom legend for currently active species
    # We create "dummy" handles just for the legend
    active_species_ids = set(item[0] for item in generation_data)
    legend_handles = []
    for s_id in active_species_ids:
        color = species_colors_registry[s_id]
        # Shorten ID for cleaner legend if it's long
        label_id = str(s_id)[:6] if isinstance(s_id, str) else str(s_id)
        patch = plt.Line2D([0], [0], marker='o', color='w', label=f"Spec {label_id}", 
                          markerfacecolor=color, markersize=10)
        legend_handles.append(patch)
    
    # Place legend outside if too many species
    if len(legend_handles) > 0:
        plt.legend(handles=legend_handles, title="Species", bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()

    # 5. Save
    filename = f"{output_dir}/gen_{generation_idx}_scatter.png"
    plt.savefig(filename)
    plt.close()
    
    return filename
