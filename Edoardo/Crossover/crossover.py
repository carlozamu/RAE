import numpy as np
import random
from Genome.agent_genome import AgentGenome

class Crossover:
    @staticmethod
    def create_offspring(parent1_genome: AgentGenome, parent2_genome: AgentGenome) -> AgentGenome:
        """
        Creates an offspring genome by combining nodes and connections from two parents.
        Ensures metadata (start/end pointers) are preserved and topology is valid.
        """
        # 1. Determine Fitter Parent (for conflict resolution)
        # Lower fitness (Loss) is better
        if parent1_genome.fitness < parent2_genome.fitness:
            fitter_parent = parent1_genome
        else:
            fitter_parent = parent2_genome
            
        offspring_genome = AgentGenome()
        
        # --- CRITICAL FIX: Inherit Start/End Pointers ---
        # We inherit these from the fitter parent (usually they are identical across species)
        offspring_genome.start_node_innovation_number = fitter_parent.start_node_innovation_number
        offspring_genome.end_node_innovation_number = fitter_parent.end_node_innovation_number
        # ------------------------------------------------
        
        # 2. Inherit Nodes
        # Union of all node innovation numbers
        all_node_innovations = set(parent1_genome.nodes.keys()) | set(parent2_genome.nodes.keys())
        
        for node_inn in all_node_innovations:
            in_p1 = node_inn in parent1_genome.nodes
            in_p2 = node_inn in parent2_genome.nodes
            
            if in_p1 and in_p2:
                # Matching Gene: Randomly select
                chosen_node = parent1_genome.nodes[node_inn] if random.random() < 0.5 else parent2_genome.nodes[node_inn]
                offspring_genome.add_node(chosen_node.copy())
            elif in_p1 and fitter_parent == parent1_genome:
                # Disjoint/Excess from Fitter Parent
                offspring_genome.add_node(parent1_genome.nodes[node_inn].copy())
            elif in_p2 and fitter_parent == parent2_genome:
                # Disjoint/Excess from Fitter Parent
                offspring_genome.add_node(parent2_genome.nodes[node_inn].copy())
            # Note: Disjoint genes from the LESS fit parent are discarded

        # 3. Inherit Connections
        all_conn_innovations = set(parent1_genome.connections.keys()) | set(parent2_genome.connections.keys())
        
        for conn_inn in all_conn_innovations:
            in_p1 = conn_inn in parent1_genome.connections
            in_p2 = conn_inn in parent2_genome.connections
            
            selected_conn = None
            
            if in_p1 and in_p2:
                # Matching Gene: Random selection
                if random.random() < 0.5:
                    selected_conn = parent1_genome.connections[conn_inn]
                else:
                    selected_conn = parent2_genome.connections[conn_inn]
            elif in_p1 and fitter_parent == parent1_genome:
                selected_conn = parent1_genome.connections[conn_inn]
            elif in_p2 and fitter_parent == parent2_genome:
                selected_conn = parent2_genome.connections[conn_inn]
            
            # 4. Safety Check: Dangling Connections
            # Only add the connection if BOTH start and end nodes exist in the offspring
            if selected_conn:
                if (selected_conn.in_node in offspring_genome.nodes and 
                    selected_conn.out_node in offspring_genome.nodes):
                    
                    # Add connection (this creates a copy internally if using your class method, 
                    # but we want to ensure we preserve the specific attributes)
                    
                    # Create a copy
                    new_conn = selected_conn.copy()
                    
                    # Manually insert to preserve the exact innovation number key
                    offspring_genome.connections[new_conn.innovation_number] = new_conn

        # 5. Clean up Topology
        offspring_genome.remove_cycles()
        
        return offspring_genome