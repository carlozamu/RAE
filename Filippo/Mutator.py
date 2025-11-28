import random
import copy
from Filippo.AgentGenome import AgentGenome, PromptNode

class MutType:
    # Architectural (Global Topology)
    ARCH_ADD_CONN = "arch_add_connection"
    ARCH_REMOVE_CONN = "arch_remove_connection"
    ARCH_ADD_NODE = "arch_add_node"     
    ARCH_REMOVE_NODE = "arch_remove_node"
    
    # Gene Level (Content & Local Topology)
    GENE_EXPAND = "gene_expand"
    GENE_SIMPLIFY = "gene_simplify"
    GENE_REFORMULATE = "gene_reformulate"
    GENE_SPLIT = "gene_split" 

class Mutator:
    """
# Example uses:

aggressive_config = {
    "p_architectural_event": 0.8, 
    "p_mutate_node": 0.5
}
mutator = Mutator(breeder_llm, config=aggressive_config)
----------------------------------------------------------
tuning_config = {
    "gene_probs": {
        MutType.GENE_EXPAND: 0.80, # Very high chance to expand
        MutType.GENE_SIMPLIFY: 0.05,
        MutType.GENE_REFORMULATE: 0.10,
        MutType.GENE_SPLIT: 0.05
    }
}
mutator = Mutator(breeder_llm, config=tuning_config)
    """
    # 1. Define Defaults (The Baseline)
    DEFAULT_CONFIG = {
        "p_architectural_event": 0.30, # Global chance of architectural mutation per offspring
        "p_mutate_node": 0.10,         # Chance per-node of content mutation
        
        # Relative weights for Architectural mutations
        "arch_probs": {
            MutType.ARCH_ADD_NODE: 0.25,
            MutType.ARCH_REMOVE_NODE: 0.25,
            MutType.ARCH_ADD_CONN: 0.25,
            MutType.ARCH_REMOVE_CONN: 0.25,
        },
        
        # Relative weights for Gene mutations
        "gene_probs": {
            MutType.GENE_EXPAND: 0.25,
            MutType.GENE_SIMPLIFY: 0.25,
            MutType.GENE_REFORMULATE: 0.30,
            MutType.GENE_SPLIT: 0.20  # 20% chance to split if selected
        }
    }

    def __init__(self, breeder_llm_client, default_config=None):
        """
        :param breeder_llm_client: Wrapper for the LLM API.
        :param default_config: Optional dict to override DEFAULT_CONFIG permanently.
        """
        self.llm = breeder_llm_client
        
        # Set the baseline configuration
        self.baseline_config = copy.deepcopy(self.DEFAULT_CONFIG)
        if default_config:
            self._recursive_update(self.baseline_config, default_config)

    def mutate(self, genome: AgentGenome, runtime_config=None) -> AgentGenome:
        """
        Main entry point. Returns a mutated CLONE of the genome.
        
        :param runtime_config: Optional dict. If provided, it merges with defaults 
                               ONLY for this single execution (e.g. for simulated annealing).
        """
        # 1. Merge Configs (Ephemeral for this run)
        if runtime_config:
            current_config = copy.deepcopy(self.baseline_config)
            self._recursive_update(current_config, runtime_config)
        else:
            current_config = self.baseline_config

        # 2. Extract active probabilities
        p_arch_event = current_config["p_architectural_event"]
        p_mutate_node = current_config["p_mutate_node"]
        
        # Build CDFs based on the (potentially dynamic) weights
        arch_cdf = self._build_cdf(current_config["arch_probs"])
        gene_cdf = self._build_cdf(current_config["gene_probs"])

        # 3. Create Offspring (Deep Copy)
        mutated_genome = genome.copy()

        # 4. Global Architectural Mutation (Single Event)
        if random.random() < p_arch_event:
            mutation_type = self._pick_from_cdf(arch_cdf)
            if mutation_type:
                self._apply_global_mutation(mutated_genome, mutation_type)

        # 5. Gene Level Mutations (Per Node Check)
        self._apply_gene_mutations(mutated_genome, p_mutate_node, gene_cdf)

        return mutated_genome

    # --- Internal Logic ---

    def _apply_global_mutation(self, genome, mutation_type):
        """Dispatches architectural mutations."""
        print(f"Applying Global Mutation: {mutation_type}")
        
        if mutation_type == MutType.ARCH_ADD_NODE:
            # Placeholder: In Phase 1, "Add Node" acts like "Append to chain" or "Insert Random"
            # In Phase 2, this manipulates the graph.
            pass
        elif mutation_type == MutType.ARCH_REMOVE_NODE:
            # Placeholder logic
            if not genome.nodes: return
            target_id = random.choice(list(genome.nodes.keys()))
            # Logic would be: find connections in/out, bridge them, remove node
            pass
        # ... Add other handlers (ADD_CONN, REMOVE_CONN) ...

    def _apply_gene_mutations(self, genome, p_mutate, gene_cdf):
        """Iterates over all nodes and applies mutations based on probability."""
        
        # CRITICAL: Snapshot values because _handle_split adds new nodes to the dict
        # We cannot iterate over genome.nodes directly while modifying it.
        nodes_snapshot = list(genome.nodes.values())

        for node in nodes_snapshot:
            # Roll dice for this specific node
            if random.random() < p_mutate:
                
                mut_type = self._pick_from_cdf(gene_cdf)

                if mut_type == MutType.GENE_SPLIT:
                    self._handle_split(genome, node)
                elif mut_type == MutType.GENE_EXPAND:
                    self._handle_content_mutation(node, "expand")
                elif mut_type == MutType.GENE_SIMPLIFY:
                    self._handle_content_mutation(node, "simplify")
                elif mut_type == MutType.GENE_REFORMULATE:
                    self._handle_content_mutation(node, "reformulate")

    # --- Specific Mutation Handlers ---

    def _handle_split(self, genome, target_node):
        """
        Hybrid Mutation: Splits one node into two sequential nodes.
        Strategy: Cell Division (Preserve A, Create B, Link A->B)
        """
        print(f"Splitting node: {target_node.name}")
        
        # 1. Ask LLM to split content
        # prompt = f"Split this instruction into two sequential steps: {target_node.instruction}"
        # inst_a_text, inst_b_text = self.llm.generate_split(prompt)
        
        # [SIMULATION]
        inst_a_text = f"(Part 1) {target_node.instruction}"
        inst_b_text = f"(Part 2) {target_node.instruction}"

        # 2. Modify Original Node (A)
        # We keep the ID and Incoming Connections intact.
        target_node.instruction = inst_a_text
        target_node.name = f"{target_node.name}_Part1"

        # 3. Create New Node (B)
        # New ID is generated automatically by __init__
        node_b = PromptNode(target_node.type, f"{target_node.name}_Part2", inst_b_text)
        genome.add_node(node_b)

        # 4. Manage Output Connections
        # We need to find all connections LEAVING the target_node and move them to B.
        conns_to_disable = []
        conns_to_create = []

        for conn in genome.connections:
            if not conn.enabled: continue
            
            # If connection goes OUT from A -> [Next]
            if conn.in_node == target_node.id:
                # We will disable this old connection
                conns_to_disable.append(conn)
                # And create a new one: B -> [Next]
                conns_to_create.append((node_b.id, conn.out_node))

        # Apply topology changes
        
        # A. Disable old outgoing connections from A
        for conn in conns_to_disable:
            conn.enabled = False

        # B. Create the Bridge: A -> B
        genome.add_connection(target_node.id, node_b.id)

        # C. Create the new outgoing connections from B
        for in_id, out_id in conns_to_create:
            genome.add_connection(in_id, out_id)

    def _handle_content_mutation(self, node, style):
        """
        Handles Expand, Simplify, and Reformulate using the LLM.
        """
        print(f"Mutating Content ({style}): {node.name}")
        
        # prompt = f"Please {style} the following instruction: {node.instruction}"
        # new_instruction = self.llm.generate_text(prompt)
        
        # [SIMULATION]
        new_instruction = f"[{style.upper()}D] {node.instruction}"
        
        if new_instruction and new_instruction.strip():
            # The setter in PromptNode automatically updates the embedding!
            node.instruction = new_instruction.strip()

    # --- Helpers ---

    def _recursive_update(self, base, update):
        """Recursively merges dictionary updates."""
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                self._recursive_update(base[k], v)
            else:
                base[k] = v

    def _build_cdf(self, probabilities):
        """Normalizes probabilities into a Cumulative Distribution Function."""
        cdf = {}
        cumulative = 0.0
        total = sum(probabilities.values())
        
        if total == 0: return {}

        for k, v in probabilities.items():
            cumulative += (v / total)
            cdf[k] = cumulative
        return cdf

    def _pick_from_cdf(self, cdf):
        """Randomly selects a key based on the CDF."""
        if not cdf: return None
        r = random.random()
        for key, cumulative_prob in cdf.items():
            if r <= cumulative_prob:
                return key
        return list(cdf.keys())[-1]