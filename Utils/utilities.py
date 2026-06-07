import gc
import os
import random
from typing import List, Dict, Any, Set, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import torch
import matplotlib.font_manager as fm

from Species.species import Species

# --- 1. Innovation Numbers Registry ---
class SemanticRegistry:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(SemanticRegistry, cls).__new__(cls)
            
            # Mathematical History (STRICTLY Arrays)
            cls._instance.registry = {}  # type Dict[int, List[np.ndarray]]
            
            # Text History (STRICTLY Strings) for debugging and telemetry
            cls._instance.prompt_history = {}  # type Dict[int, List[str]]
            
            cls._instance.next_innovation_number = 0
            
            # You may need to slightly adjust this threshold now that we are using averages.
            # Averages naturally pull scores down slightly compared to best-case single matches.
            cls._instance.similarity_threshold = 0.65 
            
        return cls._instance
    
    def reset(self):
        """Resets the registry to an empty state. Useful for testing or fresh runs."""
        self.registry.clear()
        self.prompt_history.clear()
        self.next_innovation_number = 0

    def get_or_create_innovation_number(
        self, 
        new_embedding: np.ndarray | list, 
        current_genome_node_ids: Set[int], 
        prompt_text: str,
        old_innovation_number: int = -1
    ) -> int:
        
        # 1. Prepare variables
        emb_array = np.array(new_embedding).flatten()
            
        best_id = -1
        best_sim = -1.0

        # 2. Filter registry to find only INNOVATION NUMBERS that are either:
        #    a) Not currently used in the genome (to allow reuse of old genes)
        #    b) The same as the old innovation number (to allow stable matching during mutations)
        available_ids = [
            inn_num for inn_num in self.registry.keys() 
            if inn_num not in current_genome_node_ids or inn_num == old_innovation_number
        ]
        
        # 2.1 Sort available IDs for consistent behavior (prefer older IDs if multiple have the same similarity)
        available_ids.sort()

        # 3. Find the highest compatibility across all available families (Average Linkage)
        for inn_num in available_ids:
            family_embeddings = self.registry[inn_num]
            
            # Compute similarity against EVERY member of the family
            total_sim = 0.0
            for member_emb in family_embeddings:
                sim = np.dot(emb_array, member_emb) / (np.linalg.norm(emb_array) * np.linalg.norm(member_emb))
                total_sim += sim
                
            # Calculate the exact average similarity
            avg_sim = total_sim / len(family_embeddings)
            
            # Check if this family's average is the highest we've seen
            if avg_sim > best_sim:
                best_sim = avg_sim
                best_id = inn_num

        # 4. Assignment Logic
        # We test the highest average similarity against your strict threshold
        if best_sim >= self.similarity_threshold:
            self.registry[best_id].append(emb_array)
            self.prompt_history[best_id].append(prompt_text)
            return best_id
        
        # 5. Mint a new Innovation Number
        new_id = self.next_innovation_number
        self.registry[new_id] = [emb_array]
        self.prompt_history[new_id] = [prompt_text]
        self.next_innovation_number += 1
        
        return new_id

# --- 2. Plotting Utility ---
class Plotter:
    """
    Encapsulates all plotting logic for the ERA framework.
    Includes Accuracy vs Token Usage scatter plots for efficient LLM pipeline analysis.
    """
    def __init__(self):
        self.species_colors_registry = {}
        
        # Professional, colorblind-friendly distinct palette
        self.base_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'
        ]
        
        # High-contrast baseline colors
        self.few_shot_baseline_color = "#AC9201"  # Vibrant Gold for Few-Shot
        self.zero_shot_baseline_color = "#000000" # Pure Black for Zero-Shot
        self.palette_index = 0

    def _get_species_color(self, species_id: str) -> str:
        """Retrieves or assigns a consistent, distinct color for a given species ID."""
        if species_id not in self.species_colors_registry:
            if self.palette_index < len(self.base_palette):
                self.species_colors_registry[species_id] = self.base_palette[self.palette_index]
                self.palette_index += 1
            else:
                r, g, b = random.randint(30, 200), random.randint(30, 200), random.randint(30, 200)
                self.species_colors_registry[species_id] = f'#{r:02x}{g:02x}{b:02x}'
        
        return self.species_colors_registry[species_id]

    def _parse_baseline(self, baseline_data: dict) -> Tuple[float, float]:
        """Safely extracts (accuracy, avg_tokens) whether passed as a tuple or dict."""
        return float(baseline_data.get("accuracy", 0.0)), float(baseline_data.get("avg_tokens", 0.0))

    def _parse_baseline_F(self, baseline_data: dict) -> float:
        """Safely extracts (fitness, avg_tokens) whether passed as a tuple or dict."""
        return float(baseline_data.get("fitness", 0.0))

    def plot_accuracy_vs_tokens(
        self, 
        generation_data: list, 
        zero_shot_stats: dict, 
        few_shots_stats: dict, 
        generation_idx: int, 
        output_dir="Utils/Logs/PlotsAT"
    ) -> str:
        """
        Plots Accuracy (Y-axis) vs Token Usage (X-axis).
        Includes baseline comparisons plotted as prominent stars.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Parse Baselines
        zs_acc, zs_tokens = self._parse_baseline(zero_shot_stats)
        fs_acc, fs_tokens = self._parse_baseline(few_shots_stats)

        xs, ys, colors = [], [], []

        # 2. Extract Data from Genomes
        for species in generation_data:
            if species.alive:
                species_color = self._get_species_color(str(species.id))
                for member in species.members:
                    tokens = min(1500, float(member.avg_tokens))
                    acc = float(member.accuracy)
                    
                    xs.append(tokens)
                    ys.append(acc)
                    colors.append(species_color)              

        # 3. Setup Beautiful Plot Aesthetics
        plt.figure(figsize=(11, 7), facecolor='#FAFAFA')
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')
        
        # Subdued grid lines
        ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # 4. Plot Baselines (Giant Stars so they cannot be missed)
        plt.scatter(
            [zs_tokens], [zs_acc], 
            color=self.zero_shot_baseline_color, marker='*', s=450, 
            edgecolors='white', linewidth=1.5, zorder=4
        )
        plt.scatter(
            [fs_tokens], [fs_acc], 
            color=self.few_shot_baseline_color, marker='*', s=450, 
            edgecolors='black', linewidth=1.0, zorder=3
        )

        # 5. Plot Population Scatter
        if xs:
            plt.scatter(
                xs, ys, c=colors, alpha=0.75, 
                edgecolors='white', linewidth=1.0, zorder=5
            )

        # 6. Formatting & Labels
        plt.title(f"Generation {generation_idx} Ecology: Accuracy vs. Efficiency", 
                  fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)
        
        plt.xlabel("Average Token Usage (Lower is Better)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)
        plt.ylabel("Accuracy Score (0 - 100)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)

        # Define axis limits & ticks
        plt.ylim(-5, 105)
        plt.yticks([5, 10, 15, 20, 25, 50, 75, 100])
        
        plt.xlim(90, 1010)
        plt.xticks(np.arange(100, 1550, 100)) # 100 to 1000 in steps of 100

        # 7. Professional Legend Construction
        legend_handles = []
        
        # Add Baselines to legend first
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label="Zero-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=16, 
                                       markeredgecolor='white'))
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label="Few-Shot Baseline", 
                                       markerfacecolor=self.few_shot_baseline_color, markersize=16, 
                                       markeredgecolor='black'))

        # Add Active Species
        active_species_ids = sorted([str(s.id) for s in generation_data])
        for s_key in active_species_ids:
            display_label = f"Species {s_key[:6]}" if len(s_key) > 6 else f"Species {s_key}"
            patch = plt.Line2D([0], [0], marker='o', color='w', label=display_label, 
                               markerfacecolor=self.species_colors_registry[s_key], markersize=10, 
                               markeredgecolor='white', alpha=0.8)
            legend_handles.append(patch)
        
        # Render Legend cleanly outside the main canvas
        if legend_handles:
            legend = plt.legend(
                handles=legend_handles, title="Ecology Types", 
                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                edgecolor='#E0E0E0', facecolor='white', fancybox=True
            )
            legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()

        # 8. Save Crisp High-Res Output
        filename = f"{output_dir}/gen_{generation_idx}_accuracy_vs_tokens.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
     
    def plot_fitness_vs_complexity(
        self, 
        generation_data: list[Species], 
        zero_shot_stats: dict, 
        few_shots_stats: dict, 
        generation_idx: int, 
        output_dir="Utils/Logs/PlotsFC"
    ) -> str:
        """
        Plots Fitness (Y-axis) vs Complexity (X-axis).
        Includes baseline comparisons plotted as prominent stars.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Parse Baselines
        zs_fitness  = self._parse_baseline_F(zero_shot_stats)
        fs_fitness = self._parse_baseline_F(few_shots_stats)
        zs_complexity = 1
        fs_complexity = 4

        xs, ys, colors = [], [], []

        # 2. Extract Data from Genomes
        for species in generation_data:
            if species.alive:
                species_color = self._get_species_color(str(species.id))
                for member in species.members:
                    fitness = float(member.fitness)
                    complexity = min(8, len(member.nodes))
                    
                    xs.append(complexity)
                    ys.append(fitness)
                    colors.append(species_color)
        # 3. Setup Beautiful Plot Aesthetics
        plt.figure(figsize=(11, 7), facecolor='#FAFAFA')
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')
        
        # Subdued grid lines
        ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # 4. Plot Baselines (Giant Stars so they cannot be missed)
        plt.scatter(
            [zs_complexity], [zs_fitness], 
            color=self.zero_shot_baseline_color, marker='*', s=450, 
            edgecolors='white', linewidth=1.5, zorder=4
        )
        plt.scatter(
            [fs_complexity], [fs_fitness], 
            color=self.few_shot_baseline_color, marker='*', s=450, 
            edgecolors='black', linewidth=1.0, zorder=3 
        )

        # 5. Plot Population Scatter
        if xs:
            plt.scatter(
                xs, ys, c=colors, alpha=0.75, 
                edgecolors='white', linewidth=1.0, zorder=5
            )

        # 6. Formatting & Labels
        plt.title(f"Generation {generation_idx} Ecology: Fitness vs. Complexity", 
                  fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)
        
        plt.xlabel("Complexity (nodes count)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)
        plt.ylabel("Fitness Score (0 - 100)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)

        # Define axis limits & ticks
        plt.ylim(-5, 105)
        plt.yticks([5, 10, 15, 20, 25, 50, 75, 100])
        
        plt.xlim(0.5, 8)
        plt.xticks(np.arange(1, 9, 1)) # Steps of 1 from 1 to 8

        # 7. Professional Legend Construction
        legend_handles = []
        
        # Add Baselines to legend first
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label="Zero-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=16, 
                                       markeredgecolor='white'))
        legend_handles.append(plt.Line2D([0], [0], marker='*', color='w', label="Few-Shot Baseline", 
                                       markerfacecolor=self.few_shot_baseline_color, markersize=16, 
                                       markeredgecolor='black'))

        # Add Active Species
        active_species_ids = sorted([str(s.id) for s in generation_data])
        for s_key in active_species_ids:
            display_label = f"Species {s_key[:6]}" if len(s_key) > 6 else f"Species {s_key}"
            patch = plt.Line2D([0], [0], marker='o', color='w', label=display_label, 
                               markerfacecolor=self.species_colors_registry[s_key], markersize=10, 
                               markeredgecolor='white', alpha=0.8)
            legend_handles.append(patch)
        
        # Render Legend cleanly outside the main canvas
        if legend_handles:
            legend = plt.legend(
                handles=legend_handles, title="Ecology Types", 
                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                edgecolor='#E0E0E0', facecolor='white', fancybox=True
            )
            legend.get_title().set_fontweight('bold')
        
        plt.tight_layout()

        # 8. Save Crisp High-Res Output
        filename = f"{output_dir}/gen_{generation_idx}_fitness_vs_complexity.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
     

# --- 3. History Utility ---
class HistoryTracker:
    """
    Maintains a structured history of all individuals.
    This is crucial for debugging, telemetry, and understanding the evolutionary trajectory.
    """
    def __init__(self):
        self.history: list[tuple[list[Species], dict[str, tuple[float, float]]]] = [] # List of (Species List, Baseline Stats) per generation

    def record_generation(self, species_list: List[Species], zero_shot_stats: Tuple[float, float], few_shots_stats: Tuple[float, float]):
        self.history.append((species_list, {
            "zero_shot": zero_shot_stats,
            "few_shot": few_shots_stats
        }))

# --- 4. Logging Utility ---
def log_and_print(message: str, log_file: str = "Utils/Logs/generation_logger.md"):
    print(message)
    
    # Ensure the logs directory exists
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Append the message to the markdown file
    with open(log_file, "a", encoding="utf-8") as f:
        # We strip leading newlines to avoid weird markdown formatting gaps, 
        # but keep the newline at the end for the next log.
        f.write(message.lstrip('\n') + "\n\n")

def clear_log_file(log_file: str = "Utils/Logs/generation_logger.md"):
    """
    Clears the contents of the log file before a fresh run.
    If the file or directory does not exist, it safely initializes them.
    """
    # 1. Ensure the directory exists first, just in case this function 
    # is called before log_and_print has a chance to create it.
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # 2. Open in 'w' mode. This instantly wipes all existing content.
    # We use 'pass' because we don't need to write anything; the act 
    # of opening it in 'w' mode does all the work.
    with open(log_file, "w", encoding="utf-8") as f:
        pass

def log_generation_to_markdown(species_list: List[Species], 
                               best_accuracy: float, 
                               avg_accuracy: float, 
                               zero_shot_stats: Dict[str, Any], 
                               few_shots_stats: Dict[str, Any], 
                               generation_idx: int,
                               eval_duration: float,
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
        species_accuracy_champion = max(species.members, key=lambda x: x.accuracy)
        species_avg_fitness = sum(m.fitness for m in species.members) / len(species.members)
        species_avg_accuracy = sum(m.accuracy for m in species.members) / len(species.members)

        if species_champion.fitness > global_best_fitness:
            global_best_fitness = species_champion.fitness
            
        ecology_data.append({
            "id": species.id,
            "age": species.age,
            "stagnation": species.generations_without_improvement,
            "members": len(species.members),
            "best_fit": species_champion.fitness,
            "avg_fit": species_avg_fitness,
            "best_acc": species_accuracy_champion.accuracy,
            "avg_acc": species_avg_accuracy,
            "champion": species_champion,
            "accuracy_champion": species_accuracy_champion
        })

    # 2. Build the Markdown String
    md_lines = []
    
    # --- Header ---
    md_lines.append(f"# 🧬 Generation {generation_idx} Report")
    md_lines.append(f"**Active Species:** {total_active_species} | **Global Best Fitness:** {global_best_fitness:.4f}")
    md_lines.append("\n---\n")
    
    # --- Global Performance Table ---
    md_lines.append("### 📊 Macro Performance")
    md_lines.append(f"| Engine | Accuracy | Fitness | Exec Time |")
    md_lines.append(f"| :----- | :------: | :-----: | :-------: |")
    md_lines.append(f"| **ERA (Population)**   | Best Acc: {best_accuracy:.2f}% / Avg Acc: {avg_accuracy:.2f}% | Best Fit: {global_best_fitness:.4f}   | {eval_duration:.2f}s |")
    md_lines.append(f"| **Zero-Shot Baseline** | Acc: {zero_shot_stats['accuracy']:.2f}%                       | Fit: {zero_shot_stats['fitness']:.4f} | {zero_shot_stats['execution_time']:.2f}s |")
    md_lines.append(f"| **Few-Shot Baseline**  | Acc: {few_shots_stats['accuracy']:.2f}%                       | Fit: {few_shots_stats['fitness']:.4f} | {few_shots_stats['execution_time']:.2f}s |")
    md_lines.append("\n---\n")
    
    # --- Ecology Overview Table ---
    md_lines.append("### 🌍 Ecological Overview")
    md_lines.append("| Species ID | Age | Stagnation | Members | Best Fitness | Avg Fitness |")
    md_lines.append("| :--------: | :-: | :--------: | :-----: | :----------: | :---------: |")
    
    for data in ecology_data:
        md_lines.append(f"| `{data['id']}` | {data['age']} | {data['stagnation']} | {data['members']} | {data['best_fit']:.4f} | {data['avg_fit']:.4f} | {data['best_acc']:.4f} | {data['avg_acc']:.4f} |")
    
    md_lines.append("\n---\n")
    
    # --- Species Champions (Structural Logging) ---
    md_lines.append("### 🏆 Species Champions")
    
    for data in ecology_data:
        champ = data['accuracy_champion']
        md_lines.append(f"#### Species `{data['id']}` Champion")
        md_lines.append(f"- **Genome ID:** `{champ.id}`")
        md_lines.append(f"- **Fitness:** {data['best_fit']:.4f}")
        md_lines.append(f"- **Accuracy:** {data['best_acc']:.4f}")
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

# GPU Utility
def force_cleanup():
    """Releases GPU memory."""
    print("\n🧹 Performing Memory Cleanup...")
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    print("✅ GPU Memory Released.")

