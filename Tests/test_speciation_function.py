from sentence_transformers import SentenceTransformer
from Gene.connection import Connection
from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Species.species import Species
from Utils.utilities import SemanticRegistry

# ==========================================
# 1. MOCK DATA & TARGET DISTANCES
# ==========================================
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
    "Genome_AA_1_Node": {
        "description": "Genome AA - 1 Node (BaselineBeta)",
        "start_node": "n1",
        "end_node": "n1",
        "nodes": {
            "n1": {"name": "Baseline", "instruction": "Now write the correct answer to the problem."}
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
    "Genome_CC_4_Nodes": {
        "description": "Genome C - 4 Nodes",
        "start_node": "n1",
        "end_node": "n4",
        "nodes": {
            "n1": {"name": "Work Backwards", "instruction": "Start with the final target person and work backward, listing their immediate relatives mentioned in the text."},
            "n2": {"name": "Falsification Process", "instruction": "Cross-reference this list with the starting person, explicitly eliminating logically impossible family ties."},
            "n3": {"name": "Review Reasoning", "instruction": "Evaluate the logical consistency of the information discovered until now and summarize the identified family relationships."},
            "n4": {"name": "Inquisitive Baseline", "instruction": "What specific kinship term defines their connection? Output only that single word."}
        },
        "connections": [
            {"in": "n1", "out": "n2"},
            {"in": "n1", "out": "n3"},
            {"in": "n2", "out": "n3"},
            {"in": "n3", "out": "n4"}
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
        "Genome_A_1_Node": 0.0, "Genome_AA_1_Node": 0.5, "Genome_B_3_Nodes": 4.0, 
        "Genome_C_3_Nodes": 4.0, "Genome_CC_4_Nodes": 6.5, "Genome_D_5_Nodes": 8.0, "Genome_E_7_Nodes": 9.8
    },
    "Genome_AA_1_Node": {
        "Genome_A_1_Node": 0.5, "Genome_AA_1_Node": 0.0, "Genome_B_3_Nodes": 4.0, 
        "Genome_C_3_Nodes": 4.0, "Genome_CC_4_Nodes": 6.5, "Genome_D_5_Nodes": 8.0, "Genome_E_7_Nodes": 9.8
    },
    "Genome_B_3_Nodes": {
        "Genome_A_1_Node": 4.0, "Genome_AA_1_Node": 4.0, "Genome_B_3_Nodes": 0.0, 
        "Genome_C_3_Nodes": 1.5, "Genome_CC_4_Nodes": 6.5, "Genome_D_5_Nodes": 6.0, "Genome_E_7_Nodes": 8.5
    },
    "Genome_C_3_Nodes": {
        "Genome_A_1_Node": 4.0, "Genome_AA_1_Node": 4.0, "Genome_B_3_Nodes": 1.5, 
        "Genome_C_3_Nodes": 0.0, "Genome_CC_4_Nodes": 3.0, "Genome_D_5_Nodes": 6.0, "Genome_E_7_Nodes": 8.5
    },
    "Genome_CC_4_Nodes": {
        "Genome_A_1_Node": 6.5, "Genome_AA_1_Node": 6.5, "Genome_B_3_Nodes": 3.0, 
        "Genome_C_3_Nodes": 2.5, "Genome_CC_4_Nodes": 0.0, "Genome_D_5_Nodes": 3.0, "Genome_E_7_Nodes": 6.5
    },
    "Genome_D_5_Nodes": {
        "Genome_A_1_Node": 8.0, "Genome_AA_1_Node": 8.0, "Genome_B_3_Nodes": 6.0, 
        "Genome_C_3_Nodes": 6.0, "Genome_CC_4_Nodes": 3.0, "Genome_D_5_Nodes": 0.0, "Genome_E_7_Nodes": 5.0
    },
    "Genome_E_7_Nodes": {
        "Genome_A_1_Node": 9.8, "Genome_AA_1_Node": 9.8, "Genome_B_3_Nodes": 8.5, 
        "Genome_C_3_Nodes": 8.5, "Genome_CC_4_Nodes": 6.5, "Genome_D_5_Nodes": 5.0, "Genome_E_7_Nodes": 0.0
    }
}

def get_target_distance(genome_1_id: str, genome_2_id: str) -> float:
    try:
        return target_distances[genome_1_id][genome_2_id]
    except KeyError:
        raise ValueError(f"One or both genome IDs not found in target grid: {genome_1_id}, {genome_2_id}")

# ==========================================
# 2. INITIALIZATION SCRIPT
# ==========================================
def initialize_genomes(data: dict) -> list[AgentGenome]:
    print("Loading SentenceTransformer model (BGE)...")
    # Swapped to the correct model that matches your 0.69 threshold
    model = SentenceTransformer('BAAI/bge-base-en-v1.5', device='cpu')
    
    initialized_genomes = []
    innovation_registry = SemanticRegistry() 
    # Force a reset just in case the Singleton persisted from a previous run in the same session
    innovation_registry.reset()
    
    for genome_key, genome_data in data.items():
        print(f"Initializing {genome_key}...")
        
        nodes_dict = {}
        connections_dict = {}
        
        # Instantiate Nodes 
        for node_id, node_info in genome_data["nodes"].items():
            instruction = node_info["instruction"]
            embedding = model.encode(instruction).tolist()
            
            node = PromptNode(
                name=node_info["name"],
                instruction=instruction,
                embedding=embedding,
                innovation_number=node_id # Temporary string ID (e.g., 'n1')
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

        # ---------------------------------------------------------
        # CRITICAL FIX: The Topological Cascade Repair Block
        # We must update connections alongside node keys
        # ---------------------------------------------------------
        for old_id in list(genome.nodes.keys()):
            node = genome.nodes[old_id]
            
            new_id = innovation_registry.get_or_create_innovation_number(
                new_embedding=node.embedding, 
                current_genome_node_ids=set(genome.nodes.keys()), 
                prompt_text=node.instruction,
                old_innovation_number=-1 # -1 because the old_id is currently a string ('n1')
            )
            
            if old_id != new_id:
                node.innovation_number = new_id
                
                # 1. Repair Connections pointing to this node
                for conn in list(genome.connections.values()):
                    if conn.in_node == old_id:
                        genome.add_connection(new_id, conn.out_node)
                        del genome.connections[conn.innovation_number] # Remove the old connection
                    if conn.out_node == old_id:
                        conn.update_id(new_out_node=new_id)
                        needs_update = True
                        
                        
                # 2. Repair Global Genome References
                if genome.start_node_innovation_number == old_id:
                    genome.start_node_innovation_number = new_id
                if genome.end_node_innovation_number == old_id:
                    genome.end_node_innovation_number = new_id
                    
                # 3. Swap the Node in the dict safely
                popped_node = genome.nodes.pop(old_id)
                genome.nodes[new_id] = popped_node

        initialized_genomes.append(genome)
        
    return initialized_genomes

# ==========================================
# 3. EXECUTION LOOP
# ==========================================
if __name__ == "__main__":
    genomes_list = initialize_genomes(genomes_data)
    print(f"\nSuccessfully initialized {len(genomes_list)} genomes.")
    
    genome_keys = list(genomes_data.keys())
    genomes_dict = dict(zip(genome_keys, genomes_list))

    print("\n" + "="*65)
    print(f"{'Pairing (Genome X vs Genome Y)':<35} | {'Expected':<8} | {'Actual':<8} | {'Diff':<6}")
    print("-" * 65)

    total_error = 0.0
    comparisons = 0

    for i, key1 in enumerate(genome_keys):
        for j in range(i, len(genome_keys)):
            key2 = genome_keys[j]
            
            genome1 = genomes_dict[key1]
            genome2 = genomes_dict[key2]
            
            expected_dist = get_target_distance(key1, key2)
            
            dummy_species = Species(representative=genome1, species_id=1)
            actual_dist = dummy_species.compatibility_distance(genome2)
            
            diff = abs(expected_dist - actual_dist)
            total_error += diff
            comparisons += 1
            
            name1 = key1.replace("Genome_", "")
            name2 = key2.replace("Genome_", "")
            pair_name = f"{name1} vs {name2}"
            
            print(f"{pair_name:<35} | {expected_dist:>8.2f} | {actual_dist:>8.2f} | {diff:>6.2f}")

    avg_error = total_error / comparisons
    print("-" * 65)
    print(f"Average Absolute Error: {avg_error:.3f}")
    print("=================================================================")
    print("Tuning Guide for the Species Class (Linear Output Space):")
    print("- To increase penalty for structural differences: Increase `c_nodes` and `c_edges`")
    print("- To increase penalty for cognitive/instruction differences: Increase `c_weight`")
    print(f"- Current Tuning Goal: Lower the Average Error from {avg_error:.3f} to as close to 0.0 as possible.")