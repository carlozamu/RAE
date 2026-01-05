
import json
import matplotlib.pyplot as plt
import numpy as np
import re
import seaborn as sns
import pandas as pd

def generate_plots(file_path):
    data_list = []
    
    with open(file_path, 'r') as f:
        for line in f:
            if not line.strip(): continue
            try:
                gen_data = json.loads(line)
            except:
                continue
            
            gen = gen_data.get('generation')
            # Get best individual dump detailed info if available
            best_dump = gen_data.get('best_individual_dump')
            
            # Analyze population summary provided in the 'population' list
            pop = gen_data.get('population', [])
            
            # We need to dig into the best_individual_dump to get instructions for the BEST
            # But the 'population' list typically only constitutes summary stats (fitness, num_nodes)
            # It seems 'best_individual_dump' is the only place with actual INSTRUCTIONS in this log format.
            # So we will track the "Champion's" evolution.
            
            if best_dump:
                nodes = best_dump.get('nodes', [])
                instructions = [n.get('instruction', '') for n in nodes]
                
                # Metric 1: Structured Lists (presence of "1.", "2.", etc.)
                is_structured = any(re.search(r'\b\d+\.', instr) for instr in instructions)
                
                # Metric 2: Instruction Length (avg chars)
                avg_len = sum(len(instr) for instr in instructions) / len(instructions) if instructions else 0
                
                # Metric 3: Number of Nodes
                num_nodes = len(nodes)
                
                data_list.append({
                    'Generation': gen,
                    'Fitness': best_dump.get('final_fitness', 0),
                    'Avg_Instruction_Length': avg_len,
                    'Is_Structured': 1 if is_structured else 0,
                    'Num_Nodes': num_nodes
                })

    df = pd.DataFrame(data_list)
    
    # Set style
    sns.set_theme(style="whitegrid")
    
    # PLOT 1: Emergence of Structured Reasoning
    # We smooth usage of lists to show a trend probability
    
    plt.figure(figsize=(10, 6))
    # Rolling average to show trend
    df['Structured_Rolling'] = df['Is_Structured'].rolling(window=10).mean()
    
    sns.lineplot(data=df, x='Generation', y='Structured_Rolling', linewidth=2.5, color='orange')
    plt.title('Emergence of Structured Procedures (Numbered Lists)', fontsize=16)
    plt.xlabel('Generation', fontsize=12)
    plt.ylabel('% of Champions with Structured Lists', fontsize=12)
    plt.ylim(0, 1.1)
    plt.fill_between(df['Generation'], df['Structured_Rolling'], color='orange', alpha=0.1)
    plt.savefig('Carlo/dist_plot_structure.png')
    print("Saved Carlo/dist_plot_structure.png")
    
    # PLOT 2: Semantic Complexity (Instruction Length) vs Fitness
    plt.figure(figsize=(10, 6))
    
    # Dual axis
    ax1 = plt.gca()
    sns.lineplot(data=df, x='Generation', y='Avg_Instruction_Length', ax=ax1, color='blue', label='Avg Instruction Length')
    ax1.set_ylabel('Avg Instruction Char Count', color='blue', fontsize=12)
    ax1.tick_params(axis='y', labelcolor='blue')
    
    ax2 = ax1.twinx()
    sns.lineplot(data=df, x='Generation', y='Fitness', ax=ax2, color='green', alpha=0.6, label='Best Fitness')
    ax2.set_ylabel('Best Fitness', color='green', fontsize=12)
    ax2.tick_params(axis='y', labelcolor='green')
    
    plt.title('Co-Evolution of Semantic Depth and Fitness', fontsize=16)
    plt.savefig('Carlo/dist_plot_complexity.png')
    print("Saved Carlo/dist_plot_complexity.png")

    # PLOT 3: Scatter: Instruction Length vs Fitness (Color=Gen)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(df['Avg_Instruction_Length'], df['Fitness'], 
                          c=df['Generation'], cmap='viridis', alpha=0.7, s=50)
    plt.colorbar(scatter, label='Generation')
    plt.title('Fitness Landscape: Verbosity as a Driver?', fontsize=16)
    plt.xlabel('Avg Instruction Length (chars)', fontsize=12)
    plt.ylabel('Fitness', fontsize=12)
    plt.savefig('Carlo/dist_plot_scatter.png')
    print("Saved Carlo/dist_plot_scatter.png")

if __name__ == "__main__":
    generate_plots("analysis_data.jsonl")
