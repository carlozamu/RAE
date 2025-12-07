import random
import copy
from Filippo.AgentGenome import AgentGenome, PromptNode, Connection, LM_Object

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

    def __init__(self, breeder_llm_client: LM_Object, default_config=None):
        """
        :param breeder_llm_client: Wrapper for the LLM API.
        :param default_config: Optional dict to override DEFAULT_CONFIG permanently.
        """
        self.llm = breeder_llm_client
        
        # Set the baseline configuration
        self.baseline_config = copy.deepcopy(self.DEFAULT_CONFIG)
        if default_config:
            self._recursive_update(self.baseline_config, default_config)

    # ------- Helpers Methods -------
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
    
    def _get_ancestors(self, genome: AgentGenome, target_node_id: str) -> set[str]:
        """
        Returns a set of IDs of all nodes that can reach target_node_id.
        Uses Reverse-BFS.
        """
        # 1. Build Reverse Adjacency List (Map: Node -> Parents)
        # Optimization: We build this only for the search. 
        # In a larger system, you might cache this on the genome.
        parents_map = {nid: [] for nid in genome.nodes}
        for conn in genome.connections:
            if conn.enabled:
                if conn.out_node in parents_map:
                    parents_map[conn.out_node].append(conn.in_node)
        
        # 2. Perform BFS backwards from target
        ancestors = set()
        stack = [target_node_id]
        
        while stack:
            current = stack.pop()
            # If we haven't visited this node yet (it's a new ancestor)
            for parent_id in parents_map.get(current, []):
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    stack.append(parent_id)
        
        return ancestors

    # --- Architectural Mutation Handlers ---
    def _handle_add_node(self, genome: AgentGenome):
        """
        Adds a new node by splitting an existing connection.
        """
        if not genome.connections: return

        # choose a radnom connection to split
        connection = random.choice(genome.connections)
        if not connection: return

        # get name and instructions of the in_node and out_node
        in_node = genome.nodes[connection.in_node]
        out_node = genome.nodes[connection.out_node]

        # create a new node and insert it between in_node and out_node of the chosen connection
        new_node: PromptNode = self.generate_new_node(in_node.name, in_node.instruction, out_node.name, out_node.instruction, self.llm)
        genome.add_node(new_node)
        # disable the chosen connection
        connection.enabled = False
        # add two new connections
        genome.add_connection(connection.in_node, new_node.id)
        genome.add_connection(new_node.id, connection.out_node)

    def _handle_remove_node(self, genome: AgentGenome):
        # choose a random node to remove
        node_id: str = random.choice(list(genome.nodes.keys()))
        if node_id == genome.start_node_id or node_id == genome.end_node_id:
            return # do not remove start or end nodes
        
        incoming_node_ids = []
        outgoing_node_ids = []
        
        for conn in genome.connections:
            if not conn.enabled: continue
            
            if conn.out_node == node_id:
                incoming_node_ids.append(conn.in_node)
                conn.enabled = False
            elif conn.in_node == node_id:
                outgoing_node_ids.append(conn.out_node)
                conn.enabled = False
        
        random.shuffle(incoming_node_ids)
        random.shuffle(outgoing_node_ids)

        if incoming_node_ids and outgoing_node_ids:
            # A. Determine the larger and smaller lists
            max_len = max(len(incoming_node_ids), len(outgoing_node_ids))
            
            # Loop up to the maximum count to ensure coverage
            for i in range(max_len):
                # Get input: if i is out of bounds, wrap around (modulo) 
                # or pick random to handle the "leftovers"
                in_id = incoming_node_ids[i % len(incoming_node_ids)]
                
                # Get output: same logic
                out_id = outgoing_node_ids[i % len(outgoing_node_ids)]
                
                # Add connection
                genome.add_connection(in_id, out_id)

        # 5. Delete the Node
        del genome.nodes[node_id]

    def _handle_add_connection(self, genome: AgentGenome):
        """
        Adds a new connection.
        Strategy:
        1. Pick random Source (A).
        2. Find all Ancestors of A (Nodes that can reach A).
        3. Valid Targets = All Nodes - Ancestors - Existing Targets - START.
        4. Connect A -> Random Valid Target.
        """
        # 0. If the graph is too small, skip
        if len(genome.nodes) < 3:
            return
        # 1. Candidates for Source (A): Any node except END
        # We convert values to a list to pick randomly
        possible_inputs = [n for n in genome.nodes.keys() if n != genome.end_node_id]

        # Shuffle to ensure random selection order without bias
        random.shuffle(possible_inputs)

        source_node = None
        valid_targets: list[str] = []

        # Try finding a source with at least one valid target
        # (Usually the first one works, but if the graph is fully connected, we might need to try others)
        for candidate in possible_inputs:
            
            # A. Identify invalid targets (Ancestors)
            # If B can reach A, then adding A->B creates a cycle.
            ancestor_ids = self._get_ancestors(genome, candidate)
            
            # B. Identify existing connections (Duplicates)
            existing_target_ids = set()
            for conn in genome.connections:
                if conn.in_node == candidate: # Even disabled ones count to avoid dupes
                    existing_target_ids.add(conn.out_node)
            
            # C. Filter all nodes to find valid B candidates
            # Valid B = (Not Ancestor) AND (Not Start) AND (Not Self) AND (Not Duplicate)
            candidates_for_b = []
            for node_id in genome.nodes.keys():
                if node_id == genome.start_node_id: continue   # Cannot connect to START
                if node_id == candidate: continue       # No self loops
                if node_id in ancestor_ids: continue       # No cycles
                if node_id in existing_target_ids: continue # No duplicates
                
                candidates_for_b.append(node_id)
            
            if candidates_for_b:
                source_node = candidate
                valid_targets = candidates_for_b
                break
        
        # 2. Execute Mutation if valid pair found
        if source_node and valid_targets:
            target_id = random.choice(valid_targets)
            print(f"Global: Adding connection {genome.nodes[source_node].name[:15]} -> {genome.nodes[target_id].name[:15]}")
            genome.add_connection(source_node, target_id)
        else:
            print("Global: Graph is fully saturated. No new connections possible.")
            
    def _handle_remove_connection(self, genome: AgentGenome):
        """
        Removes a random connection, ensuring graph integrity.
        Constraints:
        1. The Source node must have at least one OTHER output.
        2. The Target node must have at least one OTHER input.
        Complexity: O(E)
        """
        if not genome.connections: return
        
        # 0. If the graph is already minimal skip
        active_connections = [c for c in genome.connections if c.enabled]
        if len(active_connections) < len(genome.nodes):
            return

        # 1. Build Degree Maps (Pass 1)
        # We need to know how many active connections start/end at each node
        in_degree = {}  # Key: NodeID, Value: Count
        out_degree = {} # Key: NodeID, Value: Count

        for conn in active_connections:
            # Count Outputs (Source)
            out_degree[conn.in_node] = out_degree.get(conn.in_node, 0) + 1
            # Count Inputs (Target)
            in_degree[conn.out_node] = in_degree.get(conn.out_node, 0) + 1

        # 2. Find Valid Candidates (Pass 2)
        candidates: list[Connection] = []
        for conn in active_connections:
            # Check if removing this connection leaves nodes stranded
            source_safe = out_degree.get(conn.in_node, 0) > 1
            target_safe = in_degree.get(conn.out_node, 0) > 1
            
            if source_safe and target_safe:
                candidates.append(conn)

        # 3. Execute
        if candidates:
            target_conn = random.choice(candidates)
            print(f"Global: Removing connection {target_conn.in_node[:8]}... -> {target_conn.out_node[:8]}...")
            target_conn.enabled = False
        else:
            print("Global: No removable connections found (all are critical bridges).")

    # ------- Main Mutation Logic -------
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
    def _apply_global_mutation(self, genome: AgentGenome, mutation_type: MutType):
        """Dispatches architectural mutations."""
        print(f"Applying Global Mutation: {mutation_type}")
        
        if mutation_type == MutType.ARCH_ADD_NODE:
            self._handle_add_node(genome)
        elif mutation_type == MutType.ARCH_REMOVE_NODE:
            self._handle_remove_node(genome)
        elif mutation_type == MutType.ARCH_ADD_CONN:
            self._handle_add_connection(genome)
        elif mutation_type == MutType.ARCH_REMOVE_CONN:
            self._handle_remove_connection(genome)
     
    def _apply_gene_mutations(self, genome: AgentGenome, p_mutate, gene_cdf):
        """Iterates over all nodes and applies mutations based on probability."""
        
        # CRITICAL: Snapshot values because _handle_split adds new nodes to the dict
        # We cannot iterate over genome.nodes directly while modifying it.
        nodes_snapshot = list(genome.nodes.keys())

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

    def _handle_split(self, genome: AgentGenome, target_node: str):
        """
        Hybrid Mutation: Splits one node into two sequential nodes.
        Strategy: Cell Division (Preserve A, Create B, Link A->B)
        """
        print(f"Splitting node: {genome.nodes[target_node].name}")
        
        # 1. Ask LLM to split content
        name1, prompt1, name2,prompt2 = self.split_instructions(genome.nodes[target_node].instruction, genome.nodes[target_node].name, self.llm)

        # 2. Modify Original Node (A)
        # We keep the ID and Incoming Connections intact.
        genome.nodes[target_node].instruction = prompt1
        genome.nodes[target_node].name = name1
        
        new_gene = genome.nodes[target_node].copy()  # Copy other attributes like type, embedding, etc.
        new_gene.name = name2
        new_gene.instruction = prompt2

        genome.add_node(new_gene)

        # 3. Manage Output Connections
        # We need to find all connections LEAVING the target_node and move them to B.
        conns_to_disable: list[Connection] = []
        conns_to_create: list[str] = []

        for conn in genome.connections:
            if not conn.enabled: continue
            
            # If connection goes OUT from A -> [Next]
            if conn.in_node == target_node:
                # We will disable this old connection
                conns_to_disable.append(conn)
                # And create a new one: B -> [Next]
                conns_to_create.append((conn.out_node))

        # Apply topology changes
        
        # A. Disable old outgoing connections from A
        for conn in conns_to_disable:
            conn.enabled = False

        # B. Create the Bridge: A -> B
        genome.add_connection(target_node, new_gene.id)

        # C. Create the new outgoing connections from B
        for out_id in conns_to_create:
            genome.add_connection(new_gene.id, out_id)

    def _handle_content_mutation(self, node: PromptNode, style):
        """
        Handles Expand, Simplify, and Reformulate using the LLM. Changes only the instruction text.
        """
        print(f"Mutating Content ({style}): {node.name}")

        expand_prompt = f"Expand the following instruction to provide more detail and depth:\n\n{node.instruction}"
        simplify_prompt = f"Simplify the following instruction to make it clearer and more concise:\n\n{node.instruction}"
        reformulate_prompt = f"Reformulate the following instruction using different wording while preserving its meaning:\n\n{node.instruction}"
        
        match style:
            case "expand":
                prompt = expand_prompt
            case "simplify":
                prompt = simplify_prompt
            case "reformulate":
                prompt = reformulate_prompt

        new_instruction: str = self.llm.generate_text(prompt)
        
        if new_instruction and new_instruction.strip():
            # The setter in PromptNode automatically updates the embedding!
            node.instruction = new_instruction

    def split_instructions(original_instruction: str, original_name: str, llm: LM_Object) -> tuple[str, str, str, str]:
        """
        Uses the LLM to split an instruction into two parts.
        Returns: (prompt1, prompt2, name1, name2)
        """
        prompt = f"""Split the following instruction into two sequential parts, each with a distinct set of tasks. Provide the two new instructions along with concise names that should represent the purpose of each part.\n\n
    Original Instruction:\n{original_instruction}\n\n
    Format your response exactly as:\n
    Name1: <name for part 1>\n
    Instruction1: <instruction for part 1>\n
    Name2: <name for part 2>\n
    Instruction2: <instruction for part 2>"""
        
        response: str = llm.generate_text(prompt)

        if not response:
            print("Warning: LLM failed to split instruction. Returning original.")
            return original_name, original_instruction, original_name, original_instruction
        
        # Parse response
        name1 = response.split("Name1:")[1].split("Instruction1:")[0]
        instruction1 = response.split("Instruction1:")[1].split("Name2:")[0]
        name2 = response.split("Name2:")[1].split("Instruction2:")[0]
        instruction2 = response.split("Instruction2:")[1]
        
        return instruction1, instruction2, name1, name2

    def generate_new_node(name1: str, instruction1: str, name2: str, instruction2: str, llm: LM_Object) -> PromptNode:
        """
        Generates a new PromptNode using the LLM.
        """
        prompt = f"""A reasoning agent is composed of a set of steps, each defined by an instruction.
    Given the following sequential steps, create a new step that can logically be performed in between these two steps.
    Step 1 Name: {name1}
    Instructions Step 1: {instruction1}
    Step 2 Name: {name2}
    Instructions Step 2: {instruction2}

    You must provide a concise name for the new step and a clear instructions to be passed to the LLM as prompt.
    Format your response as follows:
    New Step Name: <name>
    Instructions: <instructions>"""

        response: str = llm.generate_text(prompt)

        # Parse the response to extract the new node name and instruction
        node_name = response.split("New Step Name:")[1].split("Instructions:")[0].strip()
        instruction = response.split("Instructions:")[1].strip()
        
        new_node = PromptNode(node_type="generic", name=node_name, instruction=instruction)
        return new_node
