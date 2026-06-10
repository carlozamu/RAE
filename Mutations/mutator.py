import random
import copy
from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Gene.connection import Connection
from Utils.utilities import SemanticRegistry
#from Utils.MarkDownLogger import md_logger
from Utils.LLM import LLM

MUTATIONS_TEMPERATURE = 0.8

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
    GENE_ADD_PERSONA = "gene_add_persona"
    GENE_INJECT_REASONING = "gene_inject_reasoning"

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
        "p_architectural_event": 0.35, # Global chance of architectural mutation per offspring
        "p_mutate_node": 0.15,         # Chance per-node of content mutation (~1/7 so that for a mature individual with ~7 nodes, we get ~1 mutation per child on average)
        
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
            MutType.GENE_REFORMULATE: 0.15,
            MutType.GENE_SPLIT: 0.10,
            MutType.GENE_ADD_PERSONA:0.10,
            MutType.GENE_INJECT_REASONING:0.15
        }
    }

    @staticmethod
    def get_dynamic_config(generation: int, node_count: int, connections_count: int) -> dict:
        """
        Calculates mutation probabilities using a strict mathematical 'Phase Shift' strategy.
        Architectural mutations act as a self-correcting thermostat centered at N=4 nodes,
        with a hard cap at N=6.
        Gene mutations shift from early Cognitive Structuring to late Linguistic Compression.
        """
        
        # --- 1. Global Rate Calculations ---
        # Architectural Decay: Cools down global topology changes over generations
        # Architectural Decay: Cools down global topology changes smoothly
        if generation < 2:
            p_arch = 0.0 
            p_gene = 0.8 
        else:
            # Shift the exponent so Gen 2 = 0
            decay_steps = generation - 2 
            
            # Starts and decays by 15% per generation
            p_arch = max(0.3, 0.75 * (0.85 ** decay_steps))
            
            # Starts and decays by 15% per generation
            p_gene = max(0.2, 0.75 * (0.85 ** decay_steps))

        # --- 2. Architectural Probabilities ---
        max_C = (node_count * (node_count - 1)) / 2.0
        min_C = node_count -1
        remove_conn = (connections_count - min_C) / max(1.0, (max_C - min_C))
         
        #remove_conn = (connections_count - (node_count - 1)) / max(1, max(0, node_count-3)*3.0)

        add_node = 0.0
        if node_count == 2:
            add_node = 0.7
        elif node_count == 3:
            add_node = 0.6
        elif node_count == 4:
            add_node = 0.5
        elif node_count >= 5:
            add_node = 0.0
        remove_node = 1.0 - add_node
        add_conn = 1.0 - remove_conn

        if node_count <= 1:
            p_add_node = 0.0
            p_remove_node = 0.0
            p_add_conn = add_conn
            p_remove_conn = remove_conn 
        else:
            p_add_node = add_node * 0.65
            p_remove_node = remove_node * 0.65
            p_add_conn = add_conn * 0.35
            p_remove_conn = remove_conn * 0.35
        
        # --- 3. Gene Probabilities (Structuring -> Compression) ---
        split = max(0.0, 0.5 - node_count * 0.1)
        others = (1.0 - split) / 5 

        return {
            "p_architectural_event": p_arch,
            "p_mutate_node": p_gene,
            
            "arch_probs": {
                MutType.ARCH_ADD_NODE:    p_add_node,
                MutType.ARCH_REMOVE_NODE: p_remove_node,
                MutType.ARCH_ADD_CONN:    p_add_conn,
                MutType.ARCH_REMOVE_CONN: p_remove_conn,
            },
            
            "gene_probs": {
                MutType.GENE_SPLIT:            split,
                MutType.GENE_INJECT_REASONING: others,
                MutType.GENE_EXPAND:           others,
                MutType.GENE_ADD_PERSONA:      others,
                MutType.GENE_REFORMULATE:      others,
                MutType.GENE_SIMPLIFY:         others
            }
        }
    def __init__(self, breeder_llm_client: LLM, default_config=None):
        """
        :param breeder_llm_client: Wrapper for the LLM API.
        :param default_config: Optional dict to override DEFAULT_CONFIG permanently.
        """
        self.llm = breeder_llm_client
        self.semantic_registry = SemanticRegistry()
        
        # Set the baseline configuration
        self.baseline_config = copy.deepcopy(self.DEFAULT_CONFIG)
        if default_config:
            self._recursive_update(self.baseline_config, default_config)

    # ------- Helpers Methods -------
    def _recursive_update(self, base: dict, update: dict):
        """Recursively merges dictionary updates."""
        for k, v in update.items():
            if isinstance(v, dict) and k in base:
                self._recursive_update(base[k], v)
            else:
                base[k] = v

    def _build_cdf(self, probabilities: dict):
        """Normalizes probabilities into a Cumulative Distribution Function."""
        cdf = {}
        cumulative = 0.0
        total = sum(probabilities.values())
        
        if total == 0: return {}

        for k, v in probabilities.items():
            cumulative += (v / total)
            cdf[k] = cumulative
        return cdf

    def _pick_from_cdf(self, cdf: dict):
        """Randomly selects a key based on the CDF."""
        if not cdf: return None
        r = random.random()
        for key, cumulative_prob in cdf.items():
            if r <= cumulative_prob:
                return key
        return list(cdf.keys())[-1]
    
    def _get_ancestors(self, genome: AgentGenome, target_node_id: int) -> set[int]:
        """
        Returns a set of IDs of all nodes that can reach target_node_id.
        Uses Reverse-DFS.
        """
        # 1. Build Reverse Adjacency List (Map: Node -> Parents)
        parents_map = {nin: [] for nin in genome.nodes.keys()}
        for conn in genome.connections.values():
            if conn.enabled:
                if conn.out_node in parents_map.keys():
                    parents_map[conn.out_node].append(conn.in_node)
        
        # 2. Perform DFS backwards from target
        ancestors = set()
        stack = [target_node_id]
        
        while stack:
            current = stack.pop()
            for parent_innovation_numbers in parents_map.get(current, []):
                if parent_innovation_numbers not in ancestors:
                    ancestors.add(parent_innovation_numbers)
                    stack.append(parent_innovation_numbers)
        
        return ancestors

    # --- Architectural Mutation Handlers ---
    async def _handle_add_node(self, genome: AgentGenome):
        """
        Adds a new node by splitting an existing connection.
        """
        if not genome.connections: return

        # choose a radnom connection to split
        connection = random.choice(list(c for c in genome.connections.values() if c.enabled))
        if not connection: return

        # get name and instructions of the in_node and out_node
        in_node = genome.nodes[connection.in_node]
        out_node = genome.nodes[connection.out_node]

        # create a new node and insert it between in_node and out_node of the chosen connection
        new_node: PromptNode = await self._generate_new_node(in_node.name, in_node.instruction, out_node.name, out_node.instruction)
        innovation_number =self.semantic_registry.get_or_create_innovation_number(new_node.embedding, set(genome.nodes.keys()), new_node.instruction)
        new_node.innovation_number = innovation_number
        genome.add_node(new_node)
        # disable the chosen connection
        connection.enabled = False
        # add two new connections
        genome.add_connection(connection.in_node, new_node.innovation_number)
        genome.add_connection(new_node.innovation_number, connection.out_node)
        #md_logger.log_event(f"""Added new node '{new_node.name}' between '{in_node.name}' and '{out_node.name}'""")

    def _handle_remove_node(self, genome: AgentGenome):
        # choose a random node to remove
        node_innovation_number: str = random.choice(list(genome.nodes.keys()))
        if node_innovation_number == genome.start_node_innovation_number or node_innovation_number == genome.end_node_innovation_number:
            return # do not remove start or end nodes
        
        incoming_node_innovation_numbers = []
        outgoing_node_innovation_numbers = []
        
        for conn in list(genome.connections.values()):            
            if conn.out_node == node_innovation_number:
                incoming_node_innovation_numbers.append(conn.in_node)
                del genome.connections[f"{conn.in_node}.{conn.out_node}"] # Remove the connection from the genome
            elif conn.in_node == node_innovation_number:
                outgoing_node_innovation_numbers.append(conn.out_node)
                del genome.connections[f"{conn.in_node}.{conn.out_node}"] # Remove the connection from the genome          
        
        random.shuffle(incoming_node_innovation_numbers)
        random.shuffle(outgoing_node_innovation_numbers)
        
        if incoming_node_innovation_numbers and outgoing_node_innovation_numbers:
            # A. Determine the larger and smaller lists
            max_len = max(len(incoming_node_innovation_numbers), len(outgoing_node_innovation_numbers))
            
            # Loop up to the maximum count to ensure coverage
            for i in range(max_len):
                # Get input: if i is out of bounds, wrap around (modulo) 
                # or pick random to handle the "leftovers"
                in_innovation_number = incoming_node_innovation_numbers[i % len(incoming_node_innovation_numbers)]
                
                # Get output: same logic
                out_innovation_number = outgoing_node_innovation_numbers[i % len(outgoing_node_innovation_numbers)]
                
                # Add connection
                genome.add_connection(in_innovation_number, out_innovation_number)

        # 5. Delete the Node
        genome.nodes.pop(node_innovation_number)
        #md_logger.log_event(f"Removed node '{node_innovation_number}' and reconnected its neighbors.")

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
        possible_inputs = [n for n in genome.nodes.keys() if n != genome.end_node_innovation_number]

        # Shuffle to ensure random selection order without bias
        random.shuffle(possible_inputs)

        source_node = None
        valid_targets: list[str] = []

        # Try finding a source with at least one valid target
        # (Usually the first one works, but if the graph is fully connected, we might need to try others)
        for candidate in possible_inputs:
            
            # A. Identify invalid targets (Ancestors)
            # If B can reach A, then adding A->B creates a cycle.
            ancestor_innovation_numbers = self._get_ancestors(genome, candidate)
            
            # B. Identify existing connections (Duplicates)
            existing_target_innovation_numbers = set()
            for conn in genome.connections.values():
                if conn.in_node == candidate: # Even disabled ones count to avoid dupes
                    existing_target_innovation_numbers.add(conn.out_node)
            
            # C. Filter all nodes to find valid B candidates
            # Valid B = (Not Ancestor) AND (Not Start) AND (Not Self) AND (Not Duplicate)
            candidates_for_b = []
            for node_innovation_number in genome.nodes.keys():
                if node_innovation_number == genome.start_node_innovation_number: continue   # Cannot connect to START
                if node_innovation_number == candidate: continue       # No self loops
                if node_innovation_number in ancestor_innovation_numbers: continue       # No cycles
                if node_innovation_number in existing_target_innovation_numbers: continue # No duplicates
                
                candidates_for_b.append(node_innovation_number)
            
            if candidates_for_b:
                source_node = candidate
                valid_targets = candidates_for_b
                break
        
        # 2. Execute Mutation if valid pair found
        if source_node and valid_targets:
            target_innovation_number = random.choice(valid_targets)
            #print(f"Global: Adding connection {genome.nodes[source_node].name} -> {genome.nodes[target_innovation_number].name}")
            genome.add_connection(source_node, target_innovation_number)
            #md_logger.log_event(f"""Added new connection from '{genome.nodes[source_node].name}' to '{genome.nodes[target_innovation_number].name}'""")
        #else:
            #print("Global: Graph is fully saturated. No new connections possible.")
            
    def _handle_remove_connection(self, genome: AgentGenome):
        """
        Removes a random connection, ensuring global graph integrity.
        Constraints: The Start Node MUST maintain at least one valid path to the End Node.
        """
        if not genome.connections: return
        
        active_connections = [c for c in genome.connections.values() if c.enabled]
        if len(active_connections) < len(genome.nodes):
            return

        # 1. Shuffle candidates so we test them randomly
        candidates = active_connections.copy()
        random.shuffle(candidates)

        # 2. Test and Revert (Global Integrity Check)
        for target_conn in candidates:
            target_conn.enabled = False
            
            # Check if Start can still reach End
            if genome.verify_all_paths_lead_to_end():
                return 
            else:
                target_conn.enabled = True
                
        # print("Global: No removable connections found (all are critical bridges).")

# ------- Main Mutation Logic -------
    async def mutate(self, genome: AgentGenome, current_generation:int=0) -> AgentGenome:
        """
        Main entry point. Returns a mutated CLONE of the genome.
        
        :param runtime_config: Optional dict. If provided, it merges with defaults 
                               ONLY for this single execution (e.g. for simulated annealing).
        """
        # 1. Get Configs
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        current_config = self.get_dynamic_config(generation=current_generation, node_count=len(genome.nodes), connections_count=len(enabled_connections))

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
                await self._apply_global_mutation(mutated_genome, mutation_type)
                mutated_genome.evaluated = False
                #md_logger.log_event(f"Applied mutation {mutation_type}")

        # 5. Gene Level Mutations (Per Node Check)
        await self._apply_gene_mutations(mutated_genome, p_mutate_node, gene_cdf)
        #md_logger.log_event(f"Applied mutations to {len(mutated_genome.nodes)} nodes")

        # 6. Check for genome consistency
        mutated_genome.remove_cycles()

        return mutated_genome

    # --- Internal Logic ---
    async def _apply_global_mutation(self, genome: AgentGenome, mutation_type: MutType):
        """Dispatches architectural mutations."""
        #print(f"Applying Global Mutation: {mutation_type}")
        
        if mutation_type == MutType.ARCH_ADD_NODE:
            await self._handle_add_node(genome)
        elif mutation_type == MutType.ARCH_REMOVE_NODE:
            self._handle_remove_node(genome)
        elif mutation_type == MutType.ARCH_ADD_CONN:
            self._handle_add_connection(genome)
        elif mutation_type == MutType.ARCH_REMOVE_CONN:
            self._handle_remove_connection(genome)
     
    async def _apply_gene_mutations(self, genome: AgentGenome, p_mutate, gene_cdf):
        """Iterates over all nodes and applies mutations based on probability."""
        
        # CRITICAL: Snapshot values because _handle_split adds new nodes to the dict
        # We cannot iterate over genome.nodes directly while modifying it.
        nodes_snapshot = list(genome.nodes.keys())

        for node in nodes_snapshot:
            # Roll dice for this specific node
            if random.random() < p_mutate:
                genome.evaluated = False
                
                mut_type = self._pick_from_cdf(gene_cdf)
                
                if mut_type == MutType.GENE_SPLIT:
                    await self._handle_split(genome, node)
                elif mut_type == MutType.GENE_EXPAND:
                    await self._handle_content_mutation(genome, node, "expand")
                elif mut_type == MutType.GENE_ADD_PERSONA:
                    await self._handle_content_mutation(genome, node, "persona")
                elif mut_type == MutType.GENE_INJECT_REASONING:
                    await self._handle_content_mutation(genome, node, "reasoning")
                elif mut_type == MutType.GENE_SIMPLIFY:
                    await self._handle_content_mutation(genome, node, "simplify")
                elif mut_type == MutType.GENE_REFORMULATE:
                    await self._handle_content_mutation(genome, node, "reformulate")

    # --- Specific Mutation Handlers ---

    async def _handle_split(self, genome: AgentGenome, target_node: int):
        """
        Hybrid Mutation: Splits one node into two sequential nodes.
        Strategy: Cell Division (Preserve A, Create B, Link A->B)
        """
        #print(f"Splitting node: {genome.nodes[target_node].name}")
        
        # 1. Ask LLM to split content
        name1, prompt1, name2, prompt2 = await self._split_instructions(genome.nodes[target_node].instruction, genome.nodes[target_node].name)

        # 2. Modify Original Node (A)
        # We keep the ID and Incoming Connections intact.
        genome.nodes[target_node].instruction = prompt1
        genome.nodes[target_node].name = name1
        genome.nodes[target_node].embedding = self.llm.get_embedding(prompt1)
        innovation_number1 =self.semantic_registry.get_or_create_innovation_number(genome.nodes[target_node].embedding, set(genome.nodes.keys()), genome.nodes[target_node].instruction, target_node)
        genome.nodes[target_node].innovation_number = innovation_number1
        if innovation_number1 != target_node:
            node = genome.nodes.pop(target_node)
            genome.add_node(node) # Update the node in the genome with new content and embedding
        
        new_gene = genome.nodes[innovation_number1].copy()
        new_gene.name = name2
        new_gene.instruction = prompt2
        new_gene.embedding = self.llm.get_embedding(prompt2)
        innovation_number2 =self.semantic_registry.get_or_create_innovation_number(new_gene.embedding, set(genome.nodes.keys()), new_gene.instruction)
        new_gene.innovation_number = innovation_number2
        genome.add_node(new_gene)

        # 3. Manage Connections
        # We need to find all connections LEAVING the target_node and move them to B.

        for conn in list(genome.connections.values()):            
            # If connection goes OUT from A -> [Next]
            if conn.in_node == target_node and innovation_number2 != target_node: # Avoid self loops in case the innovation number did't change
                genome.add_connection(innovation_number2, conn.out_node) # move to new node B
                del genome.connections[f"{target_node}.{conn.out_node}"] # Remove old connection
                #print(f"Addedd connection {innovation_number2} -> {conn.out_node} (moved from {target_node} -> {conn.out_node})")
                continue

            # If connection goes IN from [Prev] -> A
            elif conn.out_node == target_node and innovation_number1 != target_node: # Avoid self loops in case the innovation number did't change
                genome.add_connection(conn.in_node, innovation_number1) # Re-add with updated key
                del genome.connections[f"{conn.in_node}.{target_node}"] # Remove old connection
                #print(f"Addedd connection {conn.in_node} -> {innovation_number1} (moved from {conn.in_node} -> {target_node})")
                continue

        # B. Create the Bridge: A -> B
        genome.add_connection(innovation_number1, innovation_number2)
        #print(f"Added connection {innovation_number1} -> {innovation_number2} (split from {target_node})")

        # Edge case: If it was the end or start node, switch the end_innovation_number to the new node
        if target_node == genome.end_node_innovation_number:
            genome.end_node_innovation_number = innovation_number2
        if target_node == genome.start_node_innovation_number:
            genome.start_node_innovation_number = innovation_number1
        
        #print(f"Split: '{genome.nodes[innovation_number1].name}'({target_node}) --> '{genome.nodes[innovation_number1].name}'({innovation_number1}) + '{genome.nodes[innovation_number2].name}'({innovation_number2})")

    async def _handle_content_mutation(self, genome: AgentGenome, node_id: int, style: str):
        """Applies the specified mutation to the node."""

        node = genome.nodes[node_id]

        #print(f"Applying {style} mutation to node: {node.name}\nOriginal: {node.instruction}")

        expand_prompt = f"""<<start_of_turn>system
Task: Instruction Expansion.

You are an editor. Rewrite the user's instruction to make it clearer and more explicit.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL constraints and output formats.
5. Add clarity, not unnecessary length.
6. Do NOT introduce new goals.
7. Stop immediately after the rewritten instruction.
8. Do NOT add explanations, examples, or notes.
9. The output MUST be at least 20 words long.
<<end_of_turn>
<<start_of_turn>user
Original: Identify the connection between the two subjects.<end_of_turn>
<<start_of_turn>model
Detailed Version: Identify the two target subjects, analyze their shared attributes or causal links, then explicitly define the nature of their connection.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Detailed Version:"""

        expand_prompt_end = f"""<<start_of_turn>system
Task: Instruction Expansion.

You are an editor. Rewrite the user's instruction to make it clearer and more explicit.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL constraints and output formats.
5. The instruction MUST still require a single-word final answer.
6. Do NOT expand the output requirement into multiple outputs.
7. Stop immediately after the rewritten instruction.
8. Do NOT add explanations, examples, or notes.
9. The output MUST be at least 20 words long.
<<end_of_turn>
<<start_of_turn>user
Original: Identify the connection and state it in exactly one word.<end_of_turn>
<<start_of_turn>model
Detailed Version: Identify the two target subjects, analyze their shared attributes or causal links, explicitly define the nature of their connection, and state it in exactly one word.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Detailed Version:"""

        simplify_prompt = f"""<<start_of_turn>system
You are a prompt editor. You rewrite AI instructions to make them shorter and clearer.

CRITICAL RULES:
1. The input is an AI instruction (a command telling the AI what to do).
2. The output MUST be an AI instruction (a command telling the AI what to do).
3. You ONLY rewrite the instruction. You NEVER answer or execute it.
4. The output MUST contain a verb and describe the task to perform.
5. The output MUST NOT be a single word, a topic keyword, or an answer to the task.
6. Preserve the original task goal exactly.
7. Preserve ALL output format constraints.
8. If the instruction is already short, rephrase it with different words. Do NOT make it shorter.
9. The output MUST be at least 6 words long.
10. Output ONLY the rewritten instruction. No explanations.
<<end_of_turn>
<<start_of_turn>user
Original instruction: First, locate the two target subjects in the provided text. Second, trace any intermediary links or shared attributes between them. Finally, synthesize these links to deduce their exact connection.
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction: Identify the connection between the subjects by analyzing their shared attributes.
<<end_of_turn>
<<start_of_turn>user
Original instruction: Analyze the family relationship and provide the answer.
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction: Determine the family relationship and state the answer.
<<end_of_turn>
<<start_of_turn>user
Original instruction: {node.instruction}
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction:"""

        simplify_prompt_end = f"""<<start_of_turn>system
You are a prompt editor. You rewrite AI instructions to make them shorter and clearer.

CRITICAL RULES:
1. The input is an AI instruction (a command telling the AI what to do).
2. The output MUST be an AI instruction (a command telling the AI what to do).
3. You ONLY rewrite the instruction. You NEVER answer or execute it.
4. The output MUST contain a verb and describe the task to perform.
5. The output MUST NOT be a single word, a topic keyword, or an answer to the task.
6. Preserve the original task goal exactly.
7. Preserve ALL output format constraints, especially the one-word answer requirement.
8. The rewritten instruction MUST still require a single-word final answer. Do NOT remove this constraint.
9. If the instruction is already short, rephrase it with different words. Do NOT make it shorter.
10. The output MUST be at least 6 words long.

<<end_of_turn>
<<start_of_turn>user
Original instruction: First, locate the two target subjects in the provided text. Second, trace any intermediary links or shared attributes between them. Finally, synthesize these links to deduce their exact connection and state it in exactly one kinship word.
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction: Identify the connection between the subjects and state it in exactly one kinship word.
<<end_of_turn>
<<start_of_turn>user
Original instruction: Determine the exact kinship relationship and output it using exactly one word.
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction: Identify the kinship relation and reply with exactly one word.
<<end_of_turn>
<<start_of_turn>user
Original instruction: {node.instruction}
<<end_of_turn>
<<start_of_turn>model
Rewritten instruction:"""

        reformulate_prompt = f"""<<start_of_turn>system
Task: Instruction Optimization.

You are an editor. Rewrite the instruction to improve clarity for a small language model.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve the output format exactly.
5. Use short direct sentences.
6. Avoid ambiguity.
7. Prefer concrete verbs.
8. Remove redundant words.
9. Stop immediately after the rewritten instruction.
10. Do NOT add explanations, examples, or notes.
<<end_of_turn>
<<start_of_turn>user
Original: State only the single kinship word that describes the relationship.<end_of_turn>
<<start_of_turn>model
Reformulated Version: What specific kinship term defines their connection?<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Reformulated Version:"""

        reformulate_prompt_end = f"""<<start_of_turn>system
Task: Instruction Optimization.

You are an editor. Rewrite the instruction to improve clarity for a small language model.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve the output format exactly.
5. Use short direct sentences.
6. Avoid ambiguity.
7. Prefer concrete verbs.
8. Remove redundant words.
9. If the instruction requires one-word output, the rewritten instruction MUST also require one-word output.
10. Stop immediately after the rewritten instruction.
11. Do NOT add explanations, examples, or notes.
<<end_of_turn>
<<start_of_turn>user
Original: State only the single kinship word that describes the relationship.<end_of_turn>
<<start_of_turn>model
Reformulated Version: What specific kinship term defines their connection? Output only that single word.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Reformulated Version:"""

        persona_prompt = f"""<<start_of_turn>system
Task: Persona Injection.

You are an editor. Rewrite the instruction by adding a reasoning-oriented expert persona.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL constraints and output formats.
5. Use personas that reinforce analytical reasoning (e.g., careful logician, methodical analyst).
6. Do NOT introduce domain-specific assumptions that change the task.
7. The output MUST be at least 6 words long.
<<end_of_turn>
<<start_of_turn>user
Original: Identify the connection between the subjects.<end_of_turn>
<<start_of_turn>model
Mutated Version: As a careful logician, identify the connection between the subjects.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Mutated Version:"""

        persona_prompt_end = f"""<<start_of_turn>system
Task: Persona Injection.

You are an editor. Rewrite the instruction by adding a reasoning-oriented expert persona.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL constraints and output formats.
5. Use personas that reinforce analytical reasoning (e.g., careful logician, methodical analyst).
6. Do NOT introduce domain-specific assumptions that change the task.
7. The output MUST be at least 6 words long.
<<end_of_turn>
<<start_of_turn>user
Original: Identify the connection and state it in exactly one kinship term.<end_of_turn>
<<start_of_turn>model
Mutated Version: As a careful logician, identify the connection and state it in exactly one kinship term.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Mutated Version:"""

        reasoning_prompt = f"""<<start_of_turn>system
Task: Reasoning Injection.

You are an editor. Rewrite the instruction by adding a reasoning strategy to the instruction text itself.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL output constraints.
5. Add a reasoning strategy to the instruction text.
<<end_of_turn>
<<start_of_turn>user
Original: Deduce the exact relational link connecting the specified entities.<end_of_turn>
<<start_of_turn>model
Mutated Version: Deduce the exact relational link connecting the specified entities. Before answering, apply step-by-step reasoning internally to ensure accuracy.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Mutated Version:"""

        reasoning_prompt_end = f"""<<start_of_turn>system
Task: Reasoning Injection.

You are an editor. Rewrite the instruction by adding a reasoning strategy to the instruction text itself.

Rules:
1. Output ONLY the rewritten instruction.
2. NEVER execute, solve, or answer the instruction.
3. Preserve the original goal exactly.
4. Preserve ALL output constraints.
5. The instruction MUST still require a single-word final answer.
6. Add a reasoning strategy to the instruction text.
7. The instruction must encourage internal reasoning, but must NOT require the reasoning to be shown in the output.
8. The output MUST be at least 6 words long.
<<end_of_turn>
<<start_of_turn>user
Original: Deduce the exact relational link and state it in exactly one word.<end_of_turn>
<<start_of_turn>model
Mutated Version: Deduce the exact relational link. Before answering, apply step-by-step reasoning internally to ensure accuracy. State it in exactly one word.<end_of_turn>
<<start_of_turn>user
Original: {node.instruction}<end_of_turn>
<<start_of_turn>model
Mutated Version:"""


        is_end_node = (node_id == genome.end_node_innovation_number)
        # Select the prompt
        if style == "expand":
            prompt = expand_prompt_end if is_end_node else expand_prompt
        elif style == "simplify":
            prompt = simplify_prompt_end if is_end_node else simplify_prompt
        elif style == "persona":
            prompt = persona_prompt_end if is_end_node else persona_prompt
        elif style == "reasoning":
            prompt = reasoning_prompt_end if is_end_node else reasoning_prompt
        else:
            prompt = reformulate_prompt_end if is_end_node else reformulate_prompt
        
        # CRITICAL FIX: Temperature set to 0.8 to force genetic diversity
        response: str = await self.llm.generate_text(prompt, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
        
        if response and len(response.split()) > 3:
            node.instruction = response.strip()
            node.embedding = self.llm.get_embedding(response)
            innovation_number =self.semantic_registry.get_or_create_innovation_number(node.embedding, set(genome.nodes.keys()), node.instruction, node_id)
            
            if node_id != innovation_number:
                node.innovation_number = innovation_number
                #print(f"Updated innovation number for {node.name} from {node_id} to {innovation_number}")
                # If the innovation number changed, we need to update connections that point to this node
                for conn in list(genome.connections.values()):
                    if conn.in_node == node_id:
                        genome.add_connection(innovation_number, conn.out_node) # Re-add with updated key
                        del genome.connections[f"{node_id}.{conn.out_node}"]
                        #print(f"Updated connection from {node_id} -> {conn.out_node} to {innovation_number} -> {conn.out_node}")
                        continue
                    elif conn.out_node == node_id:
                        genome.add_connection(conn.in_node, innovation_number) # Re-add with updated key
                        del genome.connections[f"{conn.in_node}.{node_id}"]   
                        #print(f"Updated connection from {conn.in_node} -> {node_id} to {conn.in_node} -> {innovation_number}")
                        continue
                # If this node was the start or end node, we need to update the genome's reference
                if node_id == genome.start_node_innovation_number:
                    genome.start_node_innovation_number = innovation_number
                if node_id == genome.end_node_innovation_number:
                    genome.end_node_innovation_number = innovation_number
                node = genome.nodes.pop(node_id)
                genome.add_node(node) # Update the node in the genome list
        else:
            print(f"Failed to generate new instruction for {node.name} with style {style}. Faulty response: {response}\n")
    async def _generate_new_node(self, name1: str, inst1: str, name2: str, inst2: str) -> PromptNode:
        """
        Asks for a bridging cognitive step.
        Returns: A new PromptNode object connecting Step A and Step C.
        """
        bridge_prompt = f"""<start_of_turn>system
You are an expert cognitive architect designing reasoning pathways. Your task is to invent a logical intermediate step (Step B) that perfectly bridges the cognitive gap between Step A and Step C.

Constraint: You must output exactly ONE intermediate step using the strict format `[Step B] Name: <name> | Instr: <instruction>`. The instruction must clearly transform the outcome of Step A into the prerequisite needed for Step C. Do not include conversational filler.<end_of_turn>
<start_of_turn>user
[Step A] Name: Extract Variables | Instr: Identify the core subjects and conditions stated in the premise.
[Step C] Name: Final Deduction | Instr: Synthesize the findings into a single verifiable conclusion.<end_of_turn>
<start_of_turn>model
[Step B] Name: Formulate Dependencies | Instr: Map out how the extracted variables interact and logically depend on one another.<end_of_turn>
<start_of_turn>user
[Step A] Name: Define Objective | Instr: Clarify the exact question that needs to be answered.
[Step C] Name: Execute Solution | Instr: Calculate or derive the final answer based on the established framework.<end_of_turn>
<start_of_turn>model
[Step B] Name: Devise Strategy | Instr: Formulate a step-by-step logical plan connecting the initial objective to the required solution.<end_of_turn>
<start_of_turn>user
[Step A] Name: Read Context | Instr: Review the provided text carefully.
[Step C] Name: Filter Noise | Instr: Discard all statements that do not directly contribute to the target question.<end_of_turn>
<start_of_turn>model
[Step B] Name: Isolate Claims | Instr: Extract the specific factual claims from the text to prepare them for evaluation.<end_of_turn>
<start_of_turn>user
[Step A] Name: {name1} | Instr: {inst1}
[Step C] Name: {name2} | Instr: {inst2}<end_of_turn>
<start_of_turn>model
[Step B] Name:"""
        
        response: str = await self.llm.generate_text(bridge_prompt, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
        
        # Robust Parsing
        try:
            full_text = "Name:" + response 
            name = full_text.split("Name:")[1].split("| Instr:")[0].strip()
            instruction = full_text.split("Instr:")[1].strip()
            
            embedding = self.llm.get_embedding(instruction)
            return PromptNode(name, instruction, embedding=embedding, innovation_number=-1)
            
        except Exception as e:
            print(f"Bridge Parsing Failed: {e} with response: {response}\n")
            
            # Use a domain-agnostic cognitive instruction for the fallback to maintain fitness
            fallback_instruction = "Review the deductions made so far and explicitly map out the logical dependencies required for the next step."
            fallback_embedding = self.llm.get_embedding(fallback_instruction)
            
            return PromptNode("Logical_Bridge", fallback_instruction, embedding=fallback_embedding, innovation_number=-1)
    
    async def _split_instructions(self, original_instruction: str, original_name: str) -> tuple[str, str, str, str]:
        """
        Forces strict splitting format into two atomic cognitive steps.
        Returns: (name1, prompt1, name2, prompt2)
        """
        split_prompt = f"""<start_of_turn>system
You are an expert cognitive architect designing reasoning pathways. Your task is to split a complex instruction into two sequential, atomic sub-steps (Part 1: Preparation/Analysis, Part 2: Execution/Formatting).

Follow these rules:
1. Output exactly TWO steps using this strict format: `[Step Number]. Name: <name> | Instr: <instruction>`.
2. Each step must be a single line.
3. CRITICAL TERMINAL RULE: If the Original Instr contains a strict output format or length constraint (e.g., "CONSTRAINT: output exactly one word"), you MUST append that exact constraint verbatim to the end of the Step 2 instruction. You may also add intermediate formatting constraints to Step 1 if it helps the data flow.<end_of_turn>
<start_of_turn>user
Original Name: Isolate and Deduce
Original Instr: Discard irrelevant facts and logically deduce the hidden connection. CONSTRAINT: You must reply with exactly one word that describes the hidden connection.<end_of_turn>
<start_of_turn>model
1. Name: Isolate Facts | Instr: Review the text and explicitly discard all irrelevant statements to generate a filtered list of core facts.
2. Name: Deduce Connection | Instr: Use the isolated facts to logically deduce the final hidden connection between the subjects. CONSTRAINT: You must reply with exactly one word that describes the hidden connection.<end_of_turn>
<start_of_turn>user
Original Name: Baseline Kinship Evaluation
Original Instr: Task: State only the one kinship word (from the possible answers) that describes the family relationship between the two target individuals.<end_of_turn>
<start_of_turn>model
1. Name: Trace Kinship Tree | Instr: Read the provided text carefully and systematically map the relational lineage from the first target individual to the second.
2. Name: Format Kinship Term | Instr: Based on the mapped lineage, select the correct term from the allowed options and output only the kinship term that correctly describes the relationship.<end_of_turn>
<start_of_turn>user
Original Name: Compare
Original Instr: Task: Compare the speed of the following veichles and return only the fastest one: car, boat, truck, bicycle, plane.<end_of_turn>
<start_of_turn>model
1. Name: Understand individual speed | Instr: For each of the following veichles, write the respective top speed in km/h: car, boat, truck, bicycle, plane.
2. Name: Return fastest | Instr: Based on the top speeds, output only the name of the fastest vehicle.<end_of_turn>
<start_of_turn>user
Original Name: {original_name}
Original Instr: {original_instruction}<end_of_turn>
<start_of_turn>model
1. Name:"""
        
        response: str = await self.llm.generate_text(split_prompt, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
        
        try:
            full_text = "1. Name:" + response
            
            # Clean empty lines to prevent IndexError if the model adds newlines
            lines = [l.strip() for l in full_text.splitlines() if l.strip()]
            
            # Extract Line 1
            part1 = lines[0].split("| Instr:")
            n1 = part1[0].replace("1. Name:", "").strip()
            i1 = part1[1].strip()
            
            # Extract Line 2 (Find the first line starting with "2.")
            line2 = next(l for l in lines if l.startswith("2."))
            part2 = line2.split("| Instr:")
            n2 = part2[0].replace("2. Name:", "").strip()
            i2 = part2[1].strip()
            
            return n1, i1, n2, i2
            
        except Exception as e:
            # print(f"Split Parsing Failed: {e}. Falling back to safe split.")
            
            # SAFE FALLBACK INVERSION: 
            # Node 1 becomes a silent preparation step.
            # Node 2 retains the original instruction to preserve the terminal trap.
            prep_name = f"Prepare_{original_name.replace(' ', '_')}"
            prep_instruction = "Analyze the provided context and explicitly map out the entities involved before proceeding with the other task."
            
            return prep_name, prep_instruction, original_name, original_instruction