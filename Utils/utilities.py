import os
import random
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np

from Species.species import Species

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
class Plotter:
    """
    Encapsulates all plotting logic for the ERA framework.
    Includes Complexity vs Fitness scatter plots with a custom equidistant scale.
    """
    def __init__(self):
        # Maps species IDs to colors for consistent plotting across generations
        self.species_colors_registry = {}
        
        # Pre-define a palette of highly distinct colors for the first few species
        # This is much cleaner to read than purely random RGB values.
        self.base_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        self.palette_index = 0

    def _get_species_color(self, species_id: str) -> str:
        """Retrieves or assigns a consistent, distinct color for a given species ID."""
        if species_id not in self.species_colors_registry:
            if self.palette_index < len(self.base_palette):
                self.species_colors_registry[species_id] = self.base_palette[self.palette_index]
                self.palette_index += 1
            else:
                # Fallback to random distinct colors if we exceed the base palette
                r, g, b = random.randint(30, 200), random.randint(30, 200), random.randint(30, 200)
                self.species_colors_registry[species_id] = f'#{r:02x}{g:02x}{b:02x}'
        
        return self.species_colors_registry[species_id]

    def plot_complexity_vs_fitness(self, generation_data: List[Species], generation_idx: int, output_dir="Utils/Logs/Plots") -> str:
        """
        Plots Complexity vs Fitness using a CUSTOM EQUIDISTANT SCALE.
        Expects a list of Species objects directly from the Speciation Engine.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- A. Define the Custom Grid (0-100 Fitness) ---
        x_anchors = [0, 20, 40, 60, 80, 100]  
        y_anchors = [1, 2, 5, 10, 20, 50, 100]

        x_positions = np.arange(len(x_anchors))
        y_positions = np.arange(len(y_anchors))

        mapped_xs = []
        mapped_ys = []
        colors = []
        
        # --- B. Process Data & Assign Colors ---
        # generation_data is a List[Species]
        for species in generation_data:
            species_id_str = str(species.id)
            species_color = self._get_species_color(species_id_str)
            
            # Iterate through the actual genomes inside the species
            for member_genome in species.members:
                real_fitness = member_genome.fitness
                num_nodes = len(member_genome.nodes)
                # Count only enabled connections to measure true active complexity
                num_conns = len([c for c in member_genome.connections.values() if c.enabled])
                real_complexity = max(1, num_nodes + num_conns)
                
                # Map real values to our equidistant grid coordinates
                m_x = np.interp(real_fitness, x_anchors, x_positions)
                m_y = np.interp(real_complexity, y_anchors, y_positions)
                
                mapped_xs.append(m_x)
                mapped_ys.append(m_y)
                colors.append(species_color)

        # --- C. Create Plot ---
        plt.figure(figsize=(10, 7))
        plt.scatter(mapped_xs, mapped_ys, c=colors, alpha=0.7, edgecolors='black', linewidth=0.5, s=60)
        
        # Fake the Axis Labels to show real bounds while maintaining equidistant layout
        plt.xticks(x_positions, [str(x) for x in x_anchors])
        plt.yticks(y_positions, [str(y) for y in y_anchors])
        
        plt.xlim(-0.5, len(x_anchors) - 0.5)
        plt.ylim(-0.5, len(y_anchors) - 0.5)

        plt.title(f"Generation {generation_idx}: Complexity vs. Fitness")
        plt.xlabel("Fitness -> Higher is Better")
        plt.ylabel("Complexity (Nodes + Enabled Edges)")
        plt.grid(True, linestyle='--', alpha=0.5)

        # --- D. Legend Logic ---
        legend_handles = []
        # Extract active species IDs and sort them for a clean legend
        active_species_ids = sorted([str(s.id) for s in generation_data])
        
        for s_key in active_species_ids:
            color = self.species_colors_registry[s_key]
            
            # Safely format the display label. Truncate if it's a long UUID, leave alone if it's a short Int
            display_label = f"Species {s_key[:8]}" if len(s_key) > 8 else f"Species {s_key}"
            
            patch = plt.Line2D([0], [0], marker='o', color='w', label=display_label, 
                               markerfacecolor=color, markersize=10)
            legend_handles.append(patch)
        
        # Render legend outside the main plot area so it doesn't cover data points
        if legend_handles:
            plt.legend(handles=legend_handles, title="Active Species", bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()

        filename = f"{output_dir}/gen_{generation_idx}_scatter.png"
        plt.savefig(filename)
        plt.close()
        
        return filename
    

# --- 3. Logging Utility ---

def log_generation_to_markdown(species_list: List[Species], 
                               best_accuracy: float, 
                               avg_accuracy: float, 
                               zero_shot_stats: Dict[str, Any], 
                               few_shots_stats: Dict[str, Any], 
                               generation_idx: int,
                               log_file: str = "Utils/Logs/generation_logger.md") -> float:
    """
    Evaluates the generation's ecology, identifies champions, and appends a 
    beautifully formatted Markdown report to the logging file.
    Returns the global best fitness.
    """
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 1. Global Computations
    global_best_fitness = 0.0
    global_champion = None
    total_active_species = len(species_list)
    
    # Pre-calculate ecology stats to find the global champion
    ecology_data = []
    for species in species_list:
        if not species.members:
            continue
            
        # The members should ideally be sorted by fitness, but we use max() to be mathematically safe
        species_champion = max(species.members, key=lambda x: x.fitness)
        species_avg_fitness = sum(m.fitness for m in species.members) / len(species.members)
        
        if species_champion.fitness > global_best_fitness:
            global_best_fitness = species_champion.fitness
            global_champion = species_champion
            
        ecology_data.append({
            "id": species.id,
            "age": species.age,
            "stagnation": species.generations_without_improvement,
            "members": len(species.members),
            "best_fit": species_champion.fitness,
            "avg_fit": species_avg_fitness,
            "champion": species_champion
        })

    # 2. Build the Markdown String
    md_lines = []
    
    # --- Header ---
    md_lines.append(f"# 🧬 Generation {generation_idx} Report")
    md_lines.append(f"**Active Species:** {total_active_species} | **Global Best Fitness:** {global_best_fitness:.4f}")
    md_lines.append("\n---\n")
    
    # --- Global Performance Table ---
    md_lines.append("### 📊 Macro Performance")
    md_lines.append("| Engine | Accuracy | Fitness | Exec Time |")
    md_lines.append("| :--- | :---: | :---: | :---: |")
    md_lines.append(f"| **ERA (Population)** | {best_accuracy:.2f}% (Best) / {avg_accuracy:.2f}% (Avg) | {global_best_fitness:.4f} | - |")
    md_lines.append(f"| **Zero-Shot Baseline** | {zero_shot_stats['accuracy']:.2f}% | {zero_shot_stats['fitness']:.4f} | {zero_shot_stats['execution_time']:.2f}s |")
    md_lines.append(f"| **Few-Shot Baseline** | {few_shots_stats['accuracy']:.2f}% | {few_shots_stats['fitness']:.4f} | {few_shots_stats['execution_time']:.2f}s |")
    md_lines.append("\n---\n")
    
    # --- Ecology Overview Table ---
    md_lines.append("### 🌍 Ecological Overview")
    md_lines.append("| Species ID | Age | Stagnation | Members | Best Fitness | Avg Fitness |")
    md_lines.append("| :---: | :---: | :---: | :---: | :---: | :---: |")
    
    for data in ecology_data:
        md_lines.append(f"| `{data['id']}` | {data['age']} | {data['stagnation']} | {data['members']} | {data['best_fit']:.4f} | {data['avg_fit']:.4f} |")
    
    md_lines.append("\n---\n")
    
    # --- Species Champions (Structural Logging) ---
    md_lines.append("### 🏆 Species Champions")
    
    for data in ecology_data:
        champ = data['champion']
        md_lines.append(f"#### Species `{data['id']}` Champion")
        md_lines.append(f"- **Genome ID:** `{champ.id}`")
        md_lines.append(f"- **Fitness:** {data['best_fit']:.4f}")
        md_lines.append(f"- **Topology:** {len(champ.nodes)} Nodes, {len([c for c in champ.connections.values() if c.enabled])} Enabled Connections")
        
        # Log the actual cognitive nodes
        # Log the actual cognitive nodes in topological order
        md_lines.append("\n**Cognitive Nodes (Execution Order):**")
        
        try:
            execution_path = champ.get_execution_order()
            for step_idx, (node, dependencies) in enumerate(execution_path):
                # Safely reverse-lookup the node_id from the dictionary
                node_id = next((k for k, v in champ.nodes.items() if v == node), "?")
                
                # Flag start and end nodes for visual clarity
                marker = ""
                if node_id == champ.start_node_innovation_number:
                    marker = " **[START]**"
                elif node_id == champ.end_node_innovation_number:
                    marker = " **[END]**"
                    
                # Format dependencies for easy DAG reading
                dep_str = f" *(Waits for: {', '.join(map(str, dependencies))})*" if dependencies else ""
                    
                md_lines.append(f"> **Step {step_idx + 1}: Node {node_id}**{marker}{dep_str} -> {node.instruction}")
                
        except Exception as e:
            md_lines.append(f"> *Graph execution order could not be resolved: {e}*")
        
        md_lines.append("\n<br>\n") # Visual spacing between champions
        
    md_lines.append("\n====================================================================\n")

    # 3. Write to File
    with open(log_file, "a", encoding="utf-8") as f:
        f.write("\n".join(md_lines))
        
    return global_best_fitness
