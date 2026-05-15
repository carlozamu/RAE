from Gene.connection import Connection
from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Species.species import Species

from sentence_transformers import SentenceTransformer

# 1. Define the 5 individuals as a dictionary
genomes_data = {
    "Genome_A_1_Node": {
        "description": "Genome A - 1 Node (Baseline)",
        "start_node": "n1",
        "end_node": "n1",
        "nodes": {
            "n1": {"name": "Baseline", "instruction": "Task: State only the one kinship word (from the posible answers) that describes the family relationship."}
        },
        "connections": []
    },
    "Genome_B_3_Nodes": {
        "description": "Genome B - 3 Nodes (Sequential Chain of Thought)",
        "start_node": "n1",
        "end_node": "n3",
        "nodes": {
            "n1": {"name": "Extract Core Sentences", "instruction": "Identify the two target individuals and extract all sentences from the text that explicitly mention them."},
            "n2": {"name": "Forward Logical Chain", "instruction": "Construct a step-by-step logical chain connecting the first individual to the second based on the extracted sentences."},
            "n3": {"name": "Output Deduction", "instruction": "Based on the logical chain, output the exact family relationship word."}
        },
        "connections": [
            {"in": "n1", "out": "n2"},
            {"in": "n2", "out": "n3"}
        ]
    },
    "Genome_C_3_Nodes": {
        "description": "Genome C - 3 Nodes (Similar to B, tuned for similarity check)",
        "start_node": "n1",
        "end_node": "n3",
        "nodes": {
            "n1": {"name": "Work Backwards", "instruction": "Start with the final target person and work backward, listing their immediate relatives mentioned in the text."},
            "n2": {"name": "Falsification Process", "instruction": "Cross-reference this list with the starting person, explicitly eliminating logically impossible family ties."},
            "n3": {"name": "Inquisitive Baseline", "instruction": "What specific kinship term defines their connection? Output only that single word."}
        },
        "connections": [
            {"in": "n1", "out": "n2"},
            {"in": "n1", "out": "n3"},
            {"in": "n2", "out": "n3"}
        ]
    },
    "Genome_D_5_Nodes": {
        "description": "Genome D - 5 Nodes (Branched DAG)",
        "start_node": "n1",
        "end_node": "n5",
        "nodes": {
            "n1": {"name": "Parse Premise", "instruction": "Read the context carefully and understand the overall family structure."},
            "n2": {"name": "Analyze Person A", "instruction": "List all known immediate relatives of the first target person."},
            "n3": {"name": "Analyze Person B", "instruction": "List all known immediate relatives of the second target person."},
            "n4": {"name": "Find Intersection", "instruction": "Compare the relative lists to find the exact logical intersection between them."},
            "n5": {"name": "Formal Conclusion", "instruction": "Conclude the analysis by stating the single relational noun that logically links the two targets."}
        },
        "connections": [
            {"in": "n1", "out": "n2"},
            {"in": "n1", "out": "n3"},
            {"in": "n2", "out": "n4"},
            {"in": "n3", "out": "n4"},
            {"in": "n4", "out": "n5"}
        ]
    },
    "Genome_E_7_Nodes": {
        "description": "Genome E - 7 Nodes (Complex Analytical DAG)",
        "start_node": "n1",
        "end_node": "n7",
        "nodes": {
            "n1": {"name": "Define Objective", "instruction": "Clarify the exact family relationship that needs to be deduced."},
            "n2": {"name": "Extract Male Lineage", "instruction": "Identify all fathers, sons, uncles, and grandfathers in the text."},
            "n3": {"name": "Extract Female Lineage", "instruction": "Identify all mothers, daughters, aunts, and grandmothers in the text."},
            "n4": {"name": "Map Male Connections", "instruction": "Draw logical links between the identified male entities."},
            "n5": {"name": "Map Female Connections", "instruction": "Draw logical links between the identified female entities."},
            "n6": {"name": "Synthesize Full Tree", "instruction": "Merge the male and female logic maps into a single cohesive family tree."},
            "n7": {"name": "Terminal Definition", "instruction": "Determine the precise kinship dictionary term for the final connection and output it as a single word."}
        },
        "connections": [
            {"in": "n1", "out": "n2"},
            {"in": "n1", "out": "n3"},
            {"in": "n2", "out": "n4"},
            {"in": "n1", "out": "n4"},
            {"in": "n4", "out": "n6"},
            {"in": "n3", "out": "n5"},
            {"in": "n4", "out": "n6"},
            {"in": "n5", "out": "n6"},
            {"in": "n6", "out": "n7"}
        ]
    }
}
target_distances = {
    "Genome_A_1_Node": {
        "Genome_A_1_Node": 0.0,
        "Genome_B_3_Nodes": 4.0,  # 1 node vs 3 sequential nodes
        "Genome_C_3_Nodes": 4.0,  # 1 node vs 3 sequential nodes
        "Genome_D_5_Nodes": 8.0,  # 1 node vs 5 node DAG
        "Genome_E_7_Nodes": 9.8   # 1 node vs 7 node complex DAG (Max distance)
    },
    "Genome_B_3_Nodes": {
        "Genome_A_1_Node": 4.0,
        "Genome_B_3_Nodes": 0.0,
        "Genome_C_3_Nodes": 1.5,  # Same exact topology, purely tests embedding/cognitive distance
        "Genome_D_5_Nodes": 6.0,  # 3 seq nodes vs 5 branched nodes
        "Genome_E_7_Nodes": 8.5   # 3 seq nodes vs 7 branched nodes
    },
    "Genome_C_3_Nodes": {
        "Genome_A_1_Node": 4.0,
        "Genome_B_3_Nodes": 1.5,  # Same exact topology, purely tests embedding/cognitive distance
        "Genome_C_3_Nodes": 0.0,
        "Genome_D_5_Nodes": 6.0,  # 3 seq nodes vs 5 branched nodes
        "Genome_E_7_Nodes": 8.5   # 3 seq nodes vs 7 branched nodes
    },
    "Genome_D_5_Nodes": {
        "Genome_A_1_Node": 8.0,
        "Genome_B_3_Nodes": 6.0,
        "Genome_C_3_Nodes": 6.0,
        "Genome_D_5_Nodes": 0.0,
        "Genome_E_7_Nodes": 5.0   # 5 nodes vs 7 nodes (both are branched DAGs, closer to each other than to A)
    },
    "Genome_E_7_Nodes": {
        "Genome_A_1_Node": 9.8,
        "Genome_B_3_Nodes": 8.5,
        "Genome_C_3_Nodes": 8.5,
        "Genome_D_5_Nodes": 5.0,
        "Genome_E_7_Nodes": 0.0
    }
}

# Helper function to easily query the grid safely
def get_target_distance(genome_1_id: str, genome_2_id: str) -> float:
    """Returns the a-priori target distance between two genomes."""
    try:
        return target_distances[genome_1_id][genome_2_id]
    except KeyError:
        raise ValueError(f"One or both genome IDs not found in target grid: {genome_1_id}, {genome_2_id}")
    
# 2. Initialization Script
def initialize_genomes(data: dict) -> list[AgentGenome]:
    # Load the sentence transformer model
    print("Loading SentenceTransformer model...")
    model = SentenceTransformer('all-MiniLM-L6-v2', device='cpu')
    
    initialized_genomes = []
    
    for genome_key, genome_data in data.items():
        print(f"Initializing {genome_key}...")
        
        nodes_dict = {}
        connections_dict = {}
        
        # Instantiate Nodes and calculate embeddings
        for node_id, node_info in genome_data["nodes"].items():
            instruction = node_info["instruction"]
            embedding = model.encode(instruction).tolist()
            
            node = PromptNode(
                name=node_info["name"],
                instruction=instruction,
                embedding=embedding,
                innovation_number=node_id # Using the dict key (e.g. 'n1') as the innovation number for clear referencing
            )
            nodes_dict[node_id] = node
            
        # Instantiate Connections
        for conn_info in genome_data["connections"]:
            in_node = conn_info["in"]
            out_node = conn_info["out"]
            
            connection = Connection(
                input_node_in=in_node,
                output_node_in=out_node
            )
            connections_dict[connection.innovation_number] = connection
            
        # Create AgentGenome
        genome = AgentGenome(
            nodes_dict=nodes_dict,
            connections_dict=connections_dict,
            start_node_innovation_number=genome_data["start_node"],
            end_node_innovation_number=genome_data["end_node"],
            fitness=0.0
        )
        
        initialized_genomes.append(genome)
        
    return initialized_genomes

# Run initialization
if __name__ == "__main__":
    genomes_list = initialize_genomes(genomes_data)
    print(f"\nSuccessfully initialized {len(genomes_list)} genomes.")
    
    # Map the dictionary keys to the initialized genomes
    # (Relies on Python 3.7+ preserving dictionary insertion order)
    genome_keys = list(genomes_data.keys())
    genomes_dict = dict(zip(genome_keys, genomes_list))

    print("\n" + "="*65)
    print(f"{'Pairing (Genome X vs Genome Y)':<35} | {'Expected':<8} | {'Actual':<8} | {'Diff':<6}")
    print("-" * 65)

    total_error = 0.0
    comparisons = 0

    # Iterate through all unique pairs (combinations with replacement)
    for i, key1 in enumerate(genome_keys):
        for j in range(i, len(genome_keys)):
            key2 = genome_keys[j]
            
            genome1 = genomes_dict[key1]
            genome2 = genomes_dict[key2]
            
            # 1. Get Expected Distance
            expected_dist = get_target_distance(key1, key2)
            
            # 2. Calculate Actual Distance
            # We instantiate a dummy species with genome1 as the representative
            dummy_species = Species(representative=genome1, species_id=1)
            actual_dist = dummy_species.compatibility_distance(genome2)
            
            # 3. Calculate Difference
            diff = abs(expected_dist - actual_dist)
            total_error += diff
            comparisons += 1
            
            # Format a clean pairing name (e.g., "A_1_Node vs B_3_Nodes")
            name1 = key1.replace("Genome_", "")
            name2 = key2.replace("Genome_", "")
            pair_name = f"{name1} vs {name2}"
            
            # Print formatted result
            print(f"{pair_name:<35} | {expected_dist:>8.2f} | {actual_dist:>8.2f} | {diff:>6.2f}")

    # Summary for Hyperparameter Tuning
    avg_error = total_error / comparisons
    print("-" * 65)
    print(f"Average Absolute Error: {avg_error:.3f}")
    print("=================================================================")
    print("Tuning Guide for the Species Class:")
    print("- To increase penalty for structural differences: Increase `c_nodes` and `c_edges`")
    print("- To increase penalty for cognitive/instruction differences: Increase `c_weight`")
    print(f"- Current Tuning Goal: Lower the Average Error from {avg_error:.3f} to as close to 0.0 as possible.")
