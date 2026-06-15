import gc
import os
import re
from typing import List, Dict, Any, Set, Tuple
import matplotlib.pyplot as plt
import numpy as np
import torch
import colorsys
import pickle
import glob

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
        
        # Extended, highly distinct palette (colorblind-friendly, 30 colors)
        self.base_palette = [
            '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
            '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
            '#393b79', '#843c39', '#7b4173', '#5254a3', '#bd9e39',
            '#ad494a', '#a55194', '#6baed6', '#fd8d3c', '#74c476',
            '#f4a582', '#fb9a99', '#fdbf6f', '#cab2d6', '#b15928',
            '#e31a1c', '#ff7f00', '#6a3d9a', '#ffff99', '#b8e186'
        ]
        
        # High-contrast baseline colors
        self.few_shot_baseline_color = "#AC9201"  # Kept for compatibility
        self.zero_shot_baseline_color = "#000000" # Pure Black
        self.palette_index = 0
        self.marker_size = 100  # Unified size for all scatter markers

    def _get_species_color(self, species_id: str) -> str:
        """Retrieves or assigns a maximally distinct color for a given species ID."""
        if species_id not in self.species_colors_registry:
            if self.palette_index < len(self.base_palette):
                color = self.base_palette[self.palette_index]
            else:
                # Golden-ratio conjugate stepping on the hue wheel guarantees
                # the largest possible angular distance from every previous color.
                hue = (self.palette_index * 0.618033988749895) % 1.0
                # High saturation + medium lightness avoids browns and muddy tones
                r, g, b = colorsys.hls_to_rgb(hue, 0.55, 0.9)
                color = f'#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}'
            
            self.species_colors_registry[species_id] = color
            self.palette_index += 1
        
        return self.species_colors_registry[species_id]

    def _parse_baseline(self, baseline_data: dict) -> Tuple[float, float]:
        """Safely extracts (accuracy, avg_tokens) whether passed as a tuple or dict."""
        return float(baseline_data.get("accuracy", 0.0)), float(baseline_data.get("avg_tokens", 0.0))

    def _parse_baseline_F(self, baseline_data: dict) -> float:
        """Safely extracts fitness whether passed as a tuple or dict."""
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
        Includes baseline comparisons plotted as prominent shapes.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Parse Baselines
        zs_acc, zs_tokens = self._parse_baseline(zero_shot_stats)
        fs_acc, fs_tokens = self._parse_baseline(few_shots_stats)

        xs, ys, colors = [], [], []

        # 2. Extract Data from Genomes (ALIVE species only)
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

        # 4. Plot Baselines (same size as population dots)
        plt.scatter(
            [zs_tokens], [zs_acc], 
            color=self.zero_shot_baseline_color, marker='s', s=self.marker_size, 
            edgecolors='white', linewidth=1.5, zorder=4
        )
        plt.scatter(
            [fs_tokens], [fs_acc], 
            color=self.zero_shot_baseline_color, marker='^', s=self.marker_size, 
            edgecolors='black', linewidth=1.0, zorder=3
        )

        # 5. Plot Population Scatter
        if xs:
            plt.scatter(
                xs, ys, c=colors, alpha=0.75, s=self.marker_size,
                edgecolors='white', linewidth=1.0, zorder=5
            )

        # 6. Formatting & Labels
        plt.title(f"Generation {generation_idx} Ecology: Accuracy vs. Efficiency", 
                  fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)
        
        plt.xlabel("Average Token Usage (Lower is Better)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)
        plt.ylabel("Accuracy Score (0 - 30)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)

        # 7. Axis limits & ticks — 0 to 30, step 5
        plt.ylim(0, 30)
        plt.yticks(np.arange(0, 31, 5))
        
        plt.xlim(90, 1010)
        plt.xticks(np.arange(100, 1550, 100))

        # 8. Professional Legend Construction
        legend_handles = []
        alive_flags = []
        
        # Baselines
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label="Zero-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=10, 
                                       markeredgecolor='white'))
        alive_flags.append(True)
        legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', label="Few-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=10, 
                                       markeredgecolor='black'))
        alive_flags.append(True)

        # All Species — alive = solid, dead = semi-transparent
        sorted_species = sorted(generation_data, key=lambda s: str(s.id))
        for species in sorted_species:
            s_key = str(species.id)
            color = self._get_species_color(s_key)
            alpha = 0.8 if species.alive else 0.3
            
            display_label = f"Species {s_key[:6]}" if len(s_key) > 6 else f"Species {s_key}"
            patch = plt.Line2D([0], [0], marker='o', color='w', label=display_label, 
                               markerfacecolor=color, markersize=10, 
                               markeredgecolor='white', alpha=alpha)
            legend_handles.append(patch)
            alive_flags.append(species.alive)
        
        # Render Legend
        if legend_handles:
            legend = plt.legend(
                handles=legend_handles, title="Ecology Types", 
                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                edgecolor='#E0E0E0', facecolor='white', fancybox=True
            )
            legend.get_title().set_fontweight('bold')
            
            # Fade legend text for dead species
            for i, text in enumerate(legend.get_texts()):
                if not alive_flags[i]:
                    text.set_alpha(0.4)
        
        plt.tight_layout()

        # 9. Save Crisp High-Res Output
        filename = f"{output_dir}/gen_{generation_idx}_accuracy_vs_tokens.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename
     
    def plot_fitness_vs_complexity(
        self, 
        generation_data: List, 
        zero_shot_stats: dict, 
        few_shots_stats: dict, 
        generation_idx: int, 
        output_dir="Utils/Logs/PlotsFC"
    ) -> str:
        """
        Plots Fitness (Y-axis) vs Complexity (X-axis).
        Includes baseline comparisons plotted as prominent shapes.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # 1. Parse Baselines
        zs_fitness = self._parse_baseline_F(zero_shot_stats)
        fs_fitness = self._parse_baseline_F(few_shots_stats)
        zs_complexity = 1
        fs_complexity = 4

        xs, ys, colors = [], [], []

        # 2. Extract Data from Genomes (ALIVE species only)
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

        # 4. Plot Baselines (same size as population dots)
        plt.scatter(
            [zs_complexity], [zs_fitness], 
            color=self.zero_shot_baseline_color, marker='s', s=self.marker_size*1.5, 
            edgecolors='white', linewidth=1.5, zorder=4
        )
        plt.scatter(
            [fs_complexity], [fs_fitness], 
            color=self.zero_shot_baseline_color, marker='^', s=self.marker_size*1.5, 
            edgecolors='black', linewidth=1.0, zorder=3 
        )

        # 5. Plot Population Scatter
        if xs:
            plt.scatter(
                xs, ys, c=colors, alpha=0.75, s=self.marker_size,
                edgecolors='white', linewidth=1.0, zorder=5
            )

        # 6. Formatting & Labels
        plt.title(f"Generation {generation_idx} Ecology: Fitness vs. Complexity", 
                  fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)
        
        plt.xlabel("Complexity (nodes count)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)
        plt.ylabel("Fitness Score (0 - 30)", fontsize=12, fontweight='medium', color='#333333', labelpad=10)

        # 7. Axis limits & ticks — 0 to 30, step 5
        plt.ylim(0, 30)
        plt.yticks(np.arange(0, 31, 5))
        
        plt.xlim(0.5, 8)
        plt.xticks(np.arange(1, 9, 1))

        # 8. Professional Legend Construction
        legend_handles = []
        alive_flags = []
        
        # Baselines
        legend_handles.append(plt.Line2D([0], [0], marker='s', color='w', label="Zero-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=10, 
                                       markeredgecolor='white'))
        alive_flags.append(True)
        legend_handles.append(plt.Line2D([0], [0], marker='^', color='w', label="Few-Shot Baseline", 
                                       markerfacecolor=self.zero_shot_baseline_color, markersize=10, 
                                       markeredgecolor='black'))
        alive_flags.append(True)

        # All Species — alive = solid, dead = semi-transparent
        sorted_species = sorted(generation_data, key=lambda s: str(s.id))
        for species in sorted_species:
            s_key = str(species.id)
            color = self._get_species_color(s_key)
            alpha = 0.8 if species.alive else 0.3
            
            display_label = f"Species {s_key[:6]}" if len(s_key) > 6 else f"Species {s_key}"
            patch = plt.Line2D([0], [0], marker='o', color='w', label=display_label, 
                               markerfacecolor=color, markersize=10, 
                               markeredgecolor='white', alpha=alpha)
            legend_handles.append(patch)
            alive_flags.append(species.alive)
        
        # Render Legend
        if legend_handles:
            legend = plt.legend(
                handles=legend_handles, title="Ecology Types", 
                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True, 
                edgecolor='#E0E0E0', facecolor='white', fancybox=True
            )
            legend.get_title().set_fontweight('bold')
            
            # Fade legend text for dead species
            for i, text in enumerate(legend.get_texts()):
                if not alive_flags[i]:
                    text.set_alpha(0.4)
        
        plt.tight_layout()

        # 9. Save Crisp High-Res Output
        filename = f"{output_dir}/gen_{generation_idx}_fitness_vs_complexity.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return filename    

    def plot_accuracy_by_species(
        self,
        generation_data: list,
        zero_shot_stats: dict,
        few_shots_stats: dict,
        generation_idx: int,
        output_dir="Utils/Logs/PlotsAS"
    ) -> str:
        """
        Per-species accuracy distribution on an equidistant categorical x-axis.
        Alive species render as floating bars (height ∝ variance, centred on median)
        with whiskers to min/max. Dead species render as a single diamond at
        their representative accuracy. Baselines occupy the first two slots.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # --- 1. Parse baselines -------------------------------------------------
        zs_acc, _ = self._parse_baseline(zero_shot_stats)
        fs_acc, _ = self._parse_baseline(few_shots_stats)

        # --- 2. Sort species & build x-axis ---------------------------------------
        sorted_species = sorted(generation_data, key=lambda s: str(s.id))
        n_species = len(sorted_species)

        # 0 = ZS, 1 = FS, 2+ = species 0, 1, 2 …
        x_positions = list(range(n_species + 2))

        # Dynamic width so bars never crowd
        fig_w = max(9, n_species * 0.85 + 3)
        plt.figure(figsize=(fig_w, 7), facecolor='#FAFAFA')
        ax = plt.gca()
        ax.set_facecolor('#FAFAFA')

        ax.grid(True, linestyle='--', color='#E0E0E0', alpha=0.8, zorder=0)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color('#333333')
        ax.spines['bottom'].set_color('#333333')

        # --- 3. Plot baselines ----------------------------------------------------
        plt.scatter(
            [0], [zs_acc],
            color=self.zero_shot_baseline_color, marker='s', s=self.marker_size,
            edgecolors='white', linewidth=1.5, zorder=4
        )
        plt.scatter(
            [1], [fs_acc],
            color=self.zero_shot_baseline_color, marker='^', s=self.marker_size,
            edgecolors='black', linewidth=1.0, zorder=3
        )

        # --- 4. Plot species floating bars / dead diamonds ------------------------
        for i, species in enumerate(sorted_species):
            x = i + 2
            color = self._get_species_color(str(species.id))

            if species.alive and getattr(species, 'members', None):
                accs = [float(m.accuracy) for m in species.members]
                if not accs:
                    continue

                median = float(np.median(accs))
                q1     = float(np.percentile(accs, 25))
                q3     = float(np.percentile(accs, 75))
                min_v  = float(np.min(accs))
                max_v  = float(np.max(accs))
                iqr    = q3 - q1

                if iqr == 0 or len(accs) == 1:
                    # Collapses to a point – render as diamond
                    plt.scatter(
                        [x], [median], marker='D', c=color, s=80,
                        edgecolors='white', linewidth=1.0, alpha=0.9, zorder=5
                    )
                else:
                    # Chunky floating bar: Q1 → Q3
                    plt.bar(
                        x, iqr, bottom=q1, width=0.5,
                        color=color, edgecolor='white', linewidth=1.0,
                        alpha=0.85, zorder=4
                    )

                    # Lower whisker: Q1 down to min
                    plt.plot([x, x], [min_v, q1],
                            color=color, linewidth=1.2, alpha=0.85, zorder=3)

                    # Upper whisker: Q3 up to max
                    plt.plot([x, x], [q3, max_v],
                            color=color, linewidth=1.2, alpha=0.85, zorder=3)

                    # Short horizontal caps at min, max, and median
                    cap = 0.2
                    plt.plot([x - cap, x + cap], [min_v,  min_v],
                            color=color, linewidth=1.5, alpha=0.85, zorder=3)
                    plt.plot([x - cap, x + cap], [max_v,  max_v],
                            color=color, linewidth=1.5, alpha=0.85, zorder=3)
                    plt.plot([x - cap, x + cap], [median, median],
                            color=color, linewidth=1.5, alpha=0.95, zorder=5)
            else:
                # Dead species – single representative value
                rep_acc = 0.0
                if hasattr(species, 'representative') and species.representative is not None:
                    rep_acc = float(species.representative.accuracy)

                plt.scatter(
                    [x], [rep_acc], marker='D', c=color, s=80,
                    edgecolors='white', linewidth=1.0, alpha=0.45, zorder=5
                )

        # --- 5. X-axis labels (fade dead species) --------------------------------
        tick_labels = ['ZS', 'FS'] + [str(i) for i in range(n_species)]
        ax.set_xticks(x_positions)
        ax.set_xticklabels(tick_labels, fontsize=10, color='#333333')

        for i, label in enumerate(ax.get_xticklabels()):
            if i >= 2 and not sorted_species[i - 2].alive:
                label.set_alpha(0.35)

        # --- 6. Y-axis (accuracy 0-30, step 5) -----------------------------------
        plt.ylim(0, 30)
        plt.yticks(np.arange(0, 31, 5))
        plt.ylabel("Accuracy Score (0 - 30)", fontsize=12,
                fontweight='medium', color='#333333', labelpad=10)
        plt.xlabel("Species Index", fontsize=12,
                fontweight='medium', color='#333333', labelpad=10)
        plt.title(f"Generation {generation_idx}: Accuracy Distribution by Species",
                fontsize=16, fontweight='bold', color='#1A1A1A', pad=20)

        # --- 7. Legend (same structure, dead = semi-transparent) -----------------
        legend_handles = []
        alive_flags = []

        legend_handles.append(plt.Line2D(
            [0], [0], marker='s', color='w', label="Zero-Shot Baseline",
            markerfacecolor=self.zero_shot_baseline_color, markersize=10,
            markeredgecolor='white'))
        alive_flags.append(True)

        legend_handles.append(plt.Line2D(
            [0], [0], marker='^', color='w', label="Few-Shot Baseline",
            markerfacecolor=self.zero_shot_baseline_color, markersize=10,
            markeredgecolor='black'))
        alive_flags.append(True)

        for species in sorted_species:
            s_key = str(species.id)
            color = self._get_species_color(s_key)
            alpha = 0.8 if species.alive else 0.3

            lbl = f"Species {s_key[:6]}" if len(s_key) > 6 else f"Species {s_key}"
            patch = plt.Line2D(
                [0], [0], marker='o', color='w', label=lbl,
                markerfacecolor=color, markersize=10,
                markeredgecolor='white', alpha=alpha)
            legend_handles.append(patch)
            alive_flags.append(species.alive)

        if legend_handles:
            legend = plt.legend(
                handles=legend_handles, title="Ecology Types",
                bbox_to_anchor=(1.02, 1), loc='upper left', frameon=True,
                edgecolor='#E0E0E0', facecolor='white', fancybox=True
            )
            legend.get_title().set_fontweight('bold')
            for i, text in enumerate(legend.get_texts()):
                if not alive_flags[i]:
                    text.set_alpha(0.4)

        plt.tight_layout()
        filename = f"{output_dir}/gen_{generation_idx}_accuracy_by_species.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()

        return filename

# --- 3. History Utility ---
class HistoryTracker:
    """
    Maintains a structured history and handles rolling, multi-file checkpoints
    so evolution can be rolled back to specific generations.
    """
    def __init__(self, checkpoint_dir: str = "era_checkpoints"):
        self.checkpoint_dir = checkpoint_dir
        # Create the directory if it doesn't exist to keep your root folder clean
        if not os.path.exists(self.checkpoint_dir):
            os.makedirs(self.checkpoint_dir)
            
        self.history: list[tuple[list['Species'], dict[str, tuple[float, float]]]] = []

    def record_generation(self, species_list: List['Species'], zero_shot_stats: Dict, few_shots_stats: Dict):
        self.history.append((species_list, {
            "zero_shot": zero_shot_stats,
            "few_shot": few_shots_stats
        }))

    def save_checkpoint(self, generation_idx: int, speciation_engine: 'SpeciationEngine', zero_shot_stats: Dict, few_shots_stats: Dict):
        """
        Saves a discrete state file for the specific generation.
        """
        temp_breeder = speciation_engine.breeder
        speciation_engine.breeder = None 
        
        state = {
            "generation_idx": generation_idx,
            "speciation_engine": speciation_engine,
            "zero_shot_stats": zero_shot_stats,
            "few_shots_stats": few_shots_stats,
            "history": self.history
        }
        
        # Name the file explicitly by its generation
        filepath = os.path.join(self.checkpoint_dir, f"era_checkpoint_gen_{generation_idx}.pkl")
        temp_file = filepath + ".tmp"
        
        with open(temp_file, 'wb') as f:
            pickle.dump(state, f)
        os.replace(temp_file, filepath)
        
        speciation_engine.breeder = temp_breeder

    def load_checkpoint(self) -> Dict[str, Any]:
        """
        Scans the checkpoint directory, finds the file with the highest generation 
        number, and loads it.
        """
        if not os.path.exists(self.checkpoint_dir):
            return None
            
        # Find all files matching the checkpoint pattern
        search_pattern = os.path.join(self.checkpoint_dir, "era_checkpoint_gen_*.pkl")
        checkpoint_files = glob.glob(search_pattern)
        
        if not checkpoint_files:
            return None
            
        # Extract the integer from the filename to safely find the true maximum
        def get_gen_number(filepath: str) -> int:
            match = re.search(r'gen_(\d+)\.pkl$', filepath)
            return int(match.group(1)) if match else -1

        # Locate the file with the highest generation integer
        latest_checkpoint = max(checkpoint_files, key=get_gen_number)
        
        print(f"📂 Found rollback checkpoints. Loading highest available state: {os.path.basename(latest_checkpoint)}")
        
        with open(latest_checkpoint, 'rb') as f:
            state = pickle.load(f)
            
        # Truncate the loaded history array to match the loaded generation.
        # This prevents phantom logs from deleted future generations from remaining in memory.
        loaded_gen = state["generation_idx"]
        self.history = state.get("history", [])[:loaded_gen]
        
        return state
    
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

def just_log(message: str, log_file: str = "few_shots_results_0.txt"):    
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

def log_generation_to_markdown(species_list: List['Species'], 
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
    alive_species = [s for s in species_list if s.alive]
    
    # Pre-calculate ecology stats to find the global champion
    ecology_data = []
    for species in alive_species:
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
    md_lines.append(f"**Active Species:** {len(alive_species)} | **Global Best Fitness:** {global_best_fitness:.4f}")
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

