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
            decay_steps = generation - 1 
            p_arch = max(0.45, 0.80 * (0.95 ** decay_steps))
            p_gene = max(0.35, 0.80 * (0.95 ** decay_steps))

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

        if len(genome.nodes) >= 5:
            return
        
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

        # ============================================================
        # EXPAND
        # ============================================================

        expand_ex_1 = "Classify the sentiment of the review."
        expand_ans_1 = "Analyze the review and determine whether its overall sentiment is positive, negative, or neutral."

        expand_ex_2 = "Identify the connection between the subjects."
        expand_ans_2 = "Determine how the two subjects are related and clearly describe the nature of their connection."

        expand_ex_3 = "Extract the company name."
        expand_ans_3 = "Locate the company mentioned in the provided information and return its name."


        # ============================================================
        # EXPAND (END NODE)
        # ============================================================

        expand_end_ex_1 = "Determine the sentiment and answer with one word."
        expand_end_ans_1 = "Analyze the review and determine its sentiment, then provide the final answer using exactly one word."

        expand_end_ex_2 = "Identify the relationship and answer with one word."
        expand_end_ans_2 = "Determine how the subjects are related and provide the final answer using exactly one word."

        expand_end_ex_3 = "Determine the category and answer with one word."
        expand_end_ans_3 = "Analyze the provided information, identify the correct category, and provide the final answer using exactly one word."


        # ============================================================
        # SIMPLIFY
        # ============================================================

        simplify_ex_1 = "Carefully examine the text and determine the overall sentiment expressed by the author."
        simplify_ans_1 = "Determine the sentiment of the text."

        simplify_ex_2 = "Analyze all available evidence to infer the exact relationship between the individuals."
        simplify_ans_2 = "Determine the relationship between the individuals."

        simplify_ex_3 = "Locate the organization referenced in the document and extract its official name."
        simplify_ans_3 = "Extract the organization name."


        # ============================================================
        # SIMPLIFY (END NODE)
        # ============================================================

        simplify_end_ex_1 = "Carefully analyze the review and express its sentiment using exactly one word."
        simplify_end_ans_1 = "Determine the sentiment and answer with exactly one word."

        simplify_end_ex_2 = "Analyze the available information to identify the exact relationship and express it using a single word."
        simplify_end_ans_2 = "Determine the relationship and answer with exactly one word."

        simplify_end_ex_3 = "Determine the correct category from the provided information and provide only a single-word response."
        simplify_end_ans_3 = "Determine the category and answer with exactly one word."


        # ============================================================
        # REPHRASE
        # ============================================================

        rephrase_ex_1 = "Determine the sentiment of the text."
        rephrase_ans_1 = "What sentiment does the text express?"

        rephrase_ex_2 = "Determine the relationship between the individuals."
        rephrase_ans_2 = "How are the individuals related to one another?"

        rephrase_ex_3 = "Extract the organization name."
        rephrase_ans_3 = "Which organization is mentioned in the information provided?"


        # ============================================================
        # REPHRASE (END NODE)
        # ============================================================

        rephrase_end_ex_1 = "Determine the sentiment and answer using exactly one word."
        rephrase_end_ans_1 = "What single word best describes the sentiment?"

        rephrase_end_ex_2 = "Determine the relationship and answer using exactly one word."
        rephrase_end_ans_2 = "Which one-word term best describes the relationship?"

        rephrase_end_ex_3 = "Determine the category and output exactly one word."
        rephrase_end_ans_3 = "What single-word category best matches the information?"


        # ============================================================
        # PERSONA
        # ============================================================

        persona_ex_1 = "Determine the sentiment of the text."
        persona_ans_1 = "As an experienced sentiment analyst, determine the sentiment of the text."

        persona_ex_2 = "Determine the relationship between the individuals."
        persona_ans_2 = "As a careful relationship analyst, determine the relationship between the individuals."

        persona_ex_3 = "Extract the organization name."
        persona_ans_3 = "As an information extraction specialist, extract the organization name."


        # ============================================================
        # PERSONA (END NODE)
        # ============================================================

        persona_end_ex_1 = "Determine the sentiment and answer with exactly one word."
        persona_end_ans_1 = "As an experienced sentiment analyst, determine the sentiment and answer with exactly one word."

        persona_end_ex_2 = "Determine the relationship and answer with exactly one word."
        persona_end_ans_2 = "As a careful relationship analyst, determine the relationship and answer with exactly one word."

        persona_end_ex_3 = "Determine the category and output exactly one word."
        persona_end_ans_3 = "As a classification expert, determine the category and output exactly one word."


        # ============================================================
        # REASONING
        # ============================================================

        reasoning_ex_1 = "Determine the sentiment of the text."
        reasoning_ans_1 = "Determine the sentiment of the text. Consider all relevant evidence before deciding on the answer."

        reasoning_ex_2 = "Determine the relationship between the individuals."
        reasoning_ans_2 = "Determine the relationship between the individuals. Carefully evaluate the available information before answering."

        reasoning_ex_3 = "Extract the organization name."
        reasoning_ans_3 = "Extract the organization name. Verify that the selected entity matches the requested information before answering."


        # ============================================================
        # REASONING (END NODE)
        # ============================================================

        reasoning_end_ex_1 = "Determine the sentiment and answer using exactly one word."
        reasoning_end_ans_1 = "Determine the sentiment. Consider all relevant evidence before deciding on the answer. Respond using exactly one word."

        reasoning_end_ex_2 = "Determine the relationship and answer using exactly one word."
        reasoning_end_ans_2 = "Determine the relationship. Carefully evaluate the available information before answering. Respond using exactly one word."

        reasoning_end_ex_3 = "Determine the category and output exactly one word."
        reasoning_end_ans_3 = "Determine the category. Verify that the selected category is supported by the information before answering. Output exactly one word."

        node = genome.nodes[node_id]
        #print(f"Applying {style} mutation to node: {node.name}\nOriginal: {node.instruction}")

        is_end_node = (node_id == genome.end_node_innovation_number)
        # Select the prompt
        if style == "expand":
            task = "Instruction Expansion. Rewrite the instruction to make it clearer, more specific, and easier to follow."

            ex_1 = expand_ex_1 if not is_end_node else expand_end_ex_1
            ans_1 = expand_ans_1 if not is_end_node else expand_end_ans_1

            ex_2 = expand_ex_2 if not is_end_node else expand_end_ex_2
            ans_2 = expand_ans_2 if not is_end_node else expand_end_ans_2

            ex_3 = expand_ex_3 if not is_end_node else expand_end_ex_3
            ans_3 = expand_ans_3 if not is_end_node else expand_end_ans_3

        elif style == "simplify":
            task = "Instruction Simplification. Rewrite the instruction to make it shorter, simpler, and easier to understand."

            ex_1 = simplify_ex_1 if not is_end_node else simplify_end_ex_1
            ans_1 = simplify_ans_1 if not is_end_node else simplify_end_ans_1

            ex_2 = simplify_ex_2 if not is_end_node else simplify_end_ex_2
            ans_2 = simplify_ans_2 if not is_end_node else simplify_end_ans_2

            ex_3 = simplify_ex_3 if not is_end_node else simplify_end_ex_3
            ans_3 = simplify_ans_3 if not is_end_node else simplify_end_ans_3

        elif style == "persona":
            task = "Persona Injection. Rewrite the instruction by adding or refining an expert persona while preserving the original task."

            ex_1 = persona_ex_1 if not is_end_node else persona_end_ex_1
            ans_1 = persona_ans_1 if not is_end_node else persona_end_ans_1

            ex_2 = persona_ex_2 if not is_end_node else persona_end_ex_2
            ans_2 = persona_ans_2 if not is_end_node else persona_end_ans_2

            ex_3 = persona_ex_3 if not is_end_node else persona_end_ex_3
            ans_3 = persona_ans_3 if not is_end_node else persona_end_ans_3

        elif style == "reasoning":
            task = "Reasoning Injection. Rewrite the instruction by adding or refining a reasoning strategy that helps solve the task more accurately."

            ex_1 = reasoning_ex_1 if not is_end_node else reasoning_end_ex_1
            ans_1 = reasoning_ans_1 if not is_end_node else reasoning_end_ans_1

            ex_2 = reasoning_ex_2 if not is_end_node else reasoning_end_ex_2
            ans_2 = reasoning_ans_2 if not is_end_node else reasoning_end_ans_2

            ex_3 = reasoning_ex_3 if not is_end_node else reasoning_end_ex_3
            ans_3 = reasoning_ans_3 if not is_end_node else reasoning_end_ans_3

        else:
            task = "Instruction Rephrasing. Rewrite the instruction using a different wording or writing style while preserving its meaning."

            ex_1 = rephrase_ex_1 if not is_end_node else rephrase_end_ex_1
            ans_1 = rephrase_ans_1 if not is_end_node else rephrase_end_ans_1

            ex_2 = rephrase_ex_2 if not is_end_node else rephrase_end_ex_2
            ans_2 = rephrase_ans_2 if not is_end_node else rephrase_end_ans_2

            ex_3 = rephrase_ex_3 if not is_end_node else rephrase_end_ex_3
            ans_3 = rephrase_ans_3 if not is_end_node else rephrase_end_ans_3

        end_rule = "5. The edited instruction must still require exactly one word as the final output." if is_end_node else ""
        template = f"""<<start_of_turn>system
Task: {task}
Rules:
1. Output only the edited instruction as a single complete, natural-sounding sentence.
2. Do not execute, solve, or answer the instruction.
3. Preserve the original goal, constraints, and output format exactly.
4. Do not introduce new requirements or objectives.
{end_rule}
<<end_of_turn>
<<start_of_turn>user
Instruction: {ex_1}<end_of_turn>
<<start_of_turn>model
{ans_1}
<<end_of_turn>
<<start_of_turn>user
Instruction: {ex_2}<end_of_turn>
<<start_of_turn>model
{ans_2}
<<end_of_turn>
<<start_of_turn>user
Instruction: {ex_3}<end_of_turn>
<<start_of_turn>model
{ans_3}
<<end_of_turn>
<<start_of_turn>user
Instruction: {node.instruction}<end_of_turn>
<<start_of_turn>model
"""
        
        # CRITICAL FIX: Temperature set to 0.8 to force genetic diversity
        response: str = await self.llm.generate_text(template, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
        
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