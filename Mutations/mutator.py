import copy

import numpy as np
import random
from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Utils.utilities import SemanticRegistry
#from Utils.MarkDownLogger import md_logger
from Utils.LLM import LLM

MUTATIONS_TEMPERATURE = 0.5

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
        Calculates Macro-Layer mutation probabilities (Cooling schedule & Topology).
        Content mutation probabilities are now calculated dynamically per-node.
        """
        
        # --- 1. Global Rate Calculations ---
        if generation < 2:
            p_arch = 0.0 
            p_gene = 0.8 
        else:
            decay_steps = max(0, generation - 1)
            p_arch = max(0.25, 0.75 * (0.95 ** decay_steps))
            p_gene = max(0.30, 0.85 * (0.95 ** decay_steps))

        # --- 2. Architectural Probabilities (Strict Capping & Sparsity) ---
        if node_count >= 5:
            p_add_node = 0.0
        else:
            p_add_node = 0.6 * (0.5 ** (max(0, node_count - 2)))
            
        p_remove_node = 1.0 - p_add_node

        if node_count <= 1:
            p_add_conn = 1.0
            p_remove_conn = 0.0
        else:
            max_C = (node_count * (node_count - 1)) / 2.0
            min_C = node_count - 1
            saturation = (connections_count - min_C) / max(1.0, (max_C - min_C))
            
            p_remove_conn = saturation * 0.75
            p_add_conn = (1.0 - saturation) * 0.25

        arch_total_nodes = p_add_node + p_remove_node
        arch_total_conns = p_add_conn + p_remove_conn
        
        arch_probs = {
            MutType.ARCH_ADD_NODE:    (p_add_node / arch_total_nodes) * 0.6,
            MutType.ARCH_REMOVE_NODE: (p_remove_node / arch_total_nodes) * 0.6,
            MutType.ARCH_ADD_CONN:    (p_add_conn / max(1e-5, arch_total_conns)) * 0.4,
            MutType.ARCH_REMOVE_CONN: (p_remove_conn / max(1e-5, arch_total_conns)) * 0.4,
        }

        return {
            "p_architectural_event": p_arch,
            "p_mutate_node": p_gene,
            "arch_probs": arch_probs,
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
        rng = np.random.default_rng()
        r = rng.random()
        for key, cumulative_prob in cdf.items():
            if r <= cumulative_prob:
                return key
        return list(cdf.keys())[-1]
    
    def _get_ancestors(self, genome: AgentGenome, target_node_id: int) -> set[int]:
        parents_map = {nin: [] for nin in genome.nodes.keys()}
        for conn in genome.connections.values():
            if conn.enabled and conn.out_node in parents_map:
                parents_map[conn.out_node].append(conn.in_node)
        
        ancestors = set()
        stack = [target_node_id]
        while stack:
            current = stack.pop()
            for parent_id in parents_map.get(current, []):
                if parent_id not in ancestors:
                    ancestors.add(parent_id)
                    stack.append(parent_id)
        return ancestors

    async def _handle_add_node(self, genome: AgentGenome):
        """
        Delegates node addition to the genome's safe edge-splitting transaction.
        """
        enabled_conns = list(genome.connections.values())
        if not enabled_conns: return

        # 1. Select a random target connection to split
        target_conn = random.choice(enabled_conns)
        in_node = genome.nodes[target_conn.in_node]
        out_node = genome.nodes[target_conn.out_node]

        # 2. Generate the bridging logic
        new_node = await self._generate_new_node(in_node.name, in_node.instruction, out_node.name, out_node.instruction)
        if new_node is None: return
        
        # 3. Assign semantic innovation number
        new_id = self.semantic_registry.get_or_create_innovation_number(
            new_node.embedding, set(genome.nodes.keys()), new_node.instruction
        ) if self.semantic_registry else random.randint(10000, 99999)
        
        new_node.innovation_number = new_id
        
        # 4. Execute atomic split (AgentGenome handles all internal wiring safely)
        genome.add_node_safely(new_node, target_conn.innovation_number)

    def _handle_remove_node(self, genome: AgentGenome):
        """
        Executes a node purge, allowing the genome to compute the Cartesian bypass.
        Implements a strict rollback if the bypass creates a dead-end.
        """
        removable_nodes = [
            nid for nid in genome.nodes.keys() 
            if nid not in (genome.start_node_innovation_number, genome.end_node_innovation_number)
        ]
        if not removable_nodes: return
        
        # We only attempt one random removal per mutation event
        node_id = random.choice(removable_nodes)
        
        # 1. Execute the safe bypass transaction
        transaction = genome.remove_node_safely(node_id)
        if not transaction["removed_node"]:
            return

        # 2. Verify DAG Invariants 
        if not genome.verify_all_paths_lead_to_end():
            # ROLLBACK: The Cartesian bypass broke reachability
            
            # A. Purge the new bypass connections
            for new_c_id in transaction["added_connections"].keys():
                genome.connections.pop(new_c_id, None)
                
            # B. Restore the original node
            genome.nodes[node_id] = transaction["removed_node"]
            
            # C. Restore the original load-bearing connections
            for old_c_id, old_conn in transaction["removed_connections"].items():
                genome.connections[old_c_id] = old_conn

    def _handle_add_connection(self, genome: AgentGenome):
        """
        Brute-force attempts to add connections, letting the genome's strict
        cycle-detection and boundary constraints filter invalid pairs.
        """
        if len(genome.nodes) < 3: return
        
        possible_inputs = list(genome.nodes.keys())
        random.shuffle(possible_inputs)

        for candidate in possible_inputs:
            possible_targets = [n for n in genome.nodes.keys() if n != candidate]
            random.shuffle(possible_targets)
            
            for target in possible_targets:
                # AgentGenome automatically rejects cycles and boundary violations
                new_conn = genome.add_connection_safely(candidate, target)
                
                if new_conn is not None:
                    # A valid connection was successfully created
                    return

    def _handle_remove_connection(self, genome: AgentGenome):
        """
        Delegates connection removal to the genome's transactional method.
        """
        active_connections = list(genome.connections.keys())
        # Minimal tree check: V = E + 1 for a single path graph.
        if len(active_connections) < len(genome.nodes): return

        random.shuffle(active_connections)
        for target_conn_id in active_connections:
            # The genome automatically verifies and reverts if reachability breaks
            success = genome.remove_connection_safely(target_conn_id)
            if success:
                return

# ------- Main Mutation Logic -------
    async def mutate(self, genome: AgentGenome, current_generation: int = 0) -> AgentGenome:
        """
        Main entry point. Returns a mutated CLONE of the genome.
        """
        enabled_connections = [c for c in genome.connections.values() if c.enabled]
        total_nodes = len(genome.nodes)
        
        current_config = self.get_dynamic_config(
            generation=current_generation, 
            node_count=total_nodes, 
            connections_count=len(enabled_connections)
        )

        p_arch_event = current_config["p_architectural_event"]
        p_mutate_node = current_config["p_mutate_node"]
        arch_cdf = self._build_cdf(current_config["arch_probs"])

        mutated_genome = genome.copy()
        rng = np.random.default_rng()

        # 1. Global Architectural Mutation (Single Event)
        if rng.random() < p_arch_event:
            mutation_type = self._pick_from_cdf(arch_cdf)
            if mutation_type:
                await self._apply_global_mutation(mutated_genome, mutation_type)
                mutated_genome.evaluated = False

        # 2. Gene Level Mutations (Evaluated and dynamically weighted per node)
        await self._apply_gene_mutations(mutated_genome, p_mutate_node, total_nodes)

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
     
    async def _apply_gene_mutations(self, genome: AgentGenome, p_mutate: float, total_nodes: int):
        """
        Iterates over all nodes and applies mutations based on a dynamically 
        calculated probability distribution specific to EACH node's string length.
        """
        nodes_snapshot = list(genome.nodes.keys())
        rng = np.random.default_rng()

        for node_id in nodes_snapshot:
            if rng.random() < p_mutate:
                node = genome.nodes[node_id]
                
                # --- Micro-Layer Distribution Calculation ---
                instruction_length_tokens = len(node.instruction) // 4
                
                # Normalize length to a [0, 1] scale (assuming 150 tokens is "highly bloated")
                bloat_factor = min(1.0, instruction_length_tokens / 150.0)
                
                # 1. Split Logic: Requires both high bloat AND graph space
                # Do not attempt to split tiny instructions (< 40 chars)
                if total_nodes >= 5 or instruction_length_tokens < 30:
                    split_prob = 0.0
                else:
                    # Approaches 35% chance as the node gets extremely bloated
                    split_prob = bloat_factor * 0.35 

                # 2. Simplify vs Expand Slider
                # At 0 bloat, Expand is 45%, Simplify is 5%.
                # At 1.0 bloat, Simplify is 50%, Expand is 0%.
                simplify_prob = 0.05 + (0.45 * bloat_factor)
                expand_prob = max(0.0, 0.45 - (0.45 * bloat_factor))

                # 3. Distribute remaining probability evenly among stylistic mutations
                remaining = max(0.0, 1.0 - (split_prob + simplify_prob + expand_prob))
                others_prob = remaining / 3.0

                node_gene_probs = {
                    MutType.GENE_SPLIT:            split_prob,
                    MutType.GENE_SIMPLIFY:         simplify_prob,
                    MutType.GENE_EXPAND:           expand_prob,
                    MutType.GENE_INJECT_REASONING: others_prob*0.35,
                    MutType.GENE_ADD_PERSONA:      others_prob*0.25,
                    MutType.GENE_REFORMULATE:      others_prob*0.40
                }
                
                # Build CDF specifically for this node and select mutation
                node_cdf = self._build_cdf(node_gene_probs)
                mut_type = self._pick_from_cdf(node_cdf)
                
                # --- Execute Mutation ---
                if mut_type == MutType.GENE_SPLIT:
                    await self._handle_split(genome, node_id)
                elif mut_type == MutType.GENE_EXPAND:
                    await self._handle_content_mutation(genome, node_id, "expand")
                elif mut_type == MutType.GENE_ADD_PERSONA:
                    await self._handle_content_mutation(genome, node_id, "persona")
                elif mut_type == MutType.GENE_INJECT_REASONING:
                    await self._handle_content_mutation(genome, node_id, "reasoning")
                elif mut_type == MutType.GENE_SIMPLIFY:
                    await self._handle_content_mutation(genome, node_id, "simplify")
                elif mut_type == MutType.GENE_REFORMULATE:
                    await self._handle_content_mutation(genome, node_id, "reformulate")

    # --- Specific Mutation Handlers ---

    async def _handle_split(self, genome: AgentGenome, target_node: int):
        """
        Hybrid Mutation: Splits one node into two sequential nodes (A -> B).
        Executes as an atomic database transaction with full graph rollback capabilities.
        """
        if len(genome.nodes) >= 5: return
        
        # 1. Ask LLM to generate the split instructions
        name1, prompt1, name2, prompt2 = await self._split_instructions(
            genome.nodes[target_node].instruction, 
            genome.nodes[target_node].name
        )

        # 2. CREATE TRANSACTION BACKUP
        # Deep copy all routing data to survive catastrophic verification failures
        backup_nodes = {k: v.copy() for k, v in genome.nodes.items()}
        backup_connections = {k: v.copy() for k, v in genome.connections.items()}
        backup_start = genome.start_node_innovation_number
        backup_end = genome.end_node_innovation_number

        try:
            # 3. Formulate Node A (Modified Original)
            node_a = genome.nodes[target_node]
            node_a.instruction = prompt1
            node_a.name = name1
            node_a.embedding = self.llm.get_embedding(prompt1)
            
            inv_a = self.semantic_registry.get_or_create_innovation_number(
                node_a.embedding, set(genome.nodes.keys()), node_a.instruction, target_node
            ) if self.semantic_registry else random.randint(10000, 99999)

            # 4. Formulate Node B (New Node)
            node_b = PromptNode(name=name2, instruction=prompt2)
            node_b.embedding = self.llm.get_embedding(prompt2)
            
            inv_b = self.semantic_registry.get_or_create_innovation_number(
                node_b.embedding, set(genome.nodes.keys()), node_b.instruction
            ) if self.semantic_registry else random.randint(10000, 99999)
            
            node_b.innovation_number = inv_b

            # 5. Execute Topological Shift
            # If Node A's ID changed semantically, update its dictionary key
            if inv_a != target_node:
                node_a.innovation_number = inv_a
                genome.nodes[inv_a] = genome.nodes.pop(target_node)
            
            genome.nodes[inv_b] = node_b

            # Reroute existing connections
            conns_to_process = list(genome.connections.values())
            for conn in conns_to_process:
                if conn.in_node == target_node:
                    # Outgoing edges: Reroute to leave from Node B
                    genome.add_connection_safely(inv_b, conn.out_node)
                    genome.connections.pop(conn.innovation_number, None)
                elif conn.out_node == target_node:
                    # Incoming edges: Reroute to enter Node A (using inv_a)
                    genome.add_connection_safely(conn.in_node, inv_a)
                    genome.connections.pop(conn.innovation_number, None)

            # Build the internal bridge: A -> B
            genome.add_connection_safely(inv_a, inv_b)

            # Update Boundaries if the target was the start or end node
            if target_node == backup_start: genome.start_node_innovation_number = inv_a
            if target_node == backup_end: genome.end_node_innovation_number = inv_b

            # 6. Verify DAG Integrity
            if not genome.verify_all_paths_lead_to_end():
                raise ValueError("Split operation broke graph reachability.")
            
            genome.evaluated = False

        except Exception as e:
            # ROLLBACK: Total transaction failure. Restore exact backup state.
            genome.nodes = backup_nodes
            genome.connections = backup_connections
            genome.start_node_innovation_number = backup_start
            genome.end_node_innovation_number = backup_end

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
            task = "Persona Injection. Rewrite the instruction by adding or refining an expert persona while preserving the original task and maintaing similar tone."

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

        template = f"""<start_of_turn>system
Task: {task}
Rules:
0. NEVER execute, solve, or answer the instruction.
1. Do not introduce new requirements or objectives.
2. The scope, constraints, and output format of the original instruction must be preserved.
3. Output your answer EXCLUSIVELY as a valid JSON object using the following schema:
{{
    "edited_instruction": "The newly rewritten instruction here."
}}<end_of_turn>
<start_of_turn>user
Instruction: {ex_1}<end_of_turn>
<start_of_turn>model
```json
{{
    "edited_instruction": "{ans_1}"
}}
```<end_of_turn>
<start_of_turn>user
Instruction: {ex_2}<end_of_turn>
<start_of_turn>model
```json
{{
    "edited_instruction": "{ans_2}"
}}
```<end_of_turn>
<start_of_turn>user
Instruction: {ex_3}<end_of_turn>
<start_of_turn>model
```json
{{
    "edited_instruction": "{ans_3}"
}}
```<end_of_turn>
<start_of_turn>user
Instruction: {node.instruction}<end_of_turn>
<start_of_turn>model
"""

        response = ""
        counter = 0
        success = False
        
        while not success and counter < 7:
            response: str = await self.llm.generate_text(template, max_tokens=256, temperature=(MUTATIONS_TEMPERATURE-counter*0.1))
            counter += 1
            
            try:
                import json
                # Robust extraction: locate the JSON block even if markdown or text is present
                start_idx = response.find('{')
                end_idx = response.rfind('}') + 1
                
                if start_idx != -1 and end_idx != 0:
                    json_str = response[start_idx:end_idx]
                    data = json.loads(json_str)
                    
                    edited_instruction = data.get("edited_instruction", "").strip()
                    
                    if edited_instruction and len(edited_instruction.split()) > 3:
                        node.instruction = edited_instruction
                        node.embedding = self.llm.get_embedding(edited_instruction)
                        innovation_number = self.semantic_registry.get_or_create_innovation_number(node.embedding, set(genome.nodes.keys()), node.instruction, node_id)
                        
                        if node_id != innovation_number:
                            node.innovation_number = innovation_number
                            
                            # Re-map incoming/outgoing connections
                            for conn in list(genome.connections.values()):
                                if conn.in_node == node_id:
                                    genome.add_connection(innovation_number, conn.out_node)
                                    del genome.connections[f"{node_id}.{conn.out_node}"]
                                elif conn.out_node == node_id:
                                    genome.add_connection(conn.in_node, innovation_number)
                                    del genome.connections[f"{conn.in_node}.{node_id}"]   
                                    
                            if node_id == genome.start_node_innovation_number:
                                genome.start_node_innovation_number = innovation_number
                            if node_id == genome.end_node_innovation_number:
                                genome.end_node_innovation_number = innovation_number
                                
                            node = genome.nodes.pop(node_id)
                            genome.add_node(node) 
                            genome.evaluated = False
                        
                        success = True # Exit the while loop
                    else:
                        print(f"Attempt {counter}: Parsed instruction too short.")
                else:
                    print(f"Attempt {counter}: Failed to locate JSON brackets.")
                    
            except json.JSONDecodeError as e:
                print(f"Attempt {counter}: JSON Decode Error: {e}")
            except Exception as e:
                print(f"Attempt {counter}: Unexpected error during mutation parsing: {e}")

        if not success:
            print(f"Failed to generate valid JSON instruction for {node.name} with style {style} after 7 attempts.\nLast response: {response}")
    
    async def _generate_new_node(self, name1: str, inst1: str, name2: str, inst2: str) -> PromptNode:
        bridge_prompt = f"""<start_of_turn>system
You are an expert cognitive architect designing reasoning pathways. Your task is to invent a logical intermediate step (Step B) that bridges the cognitive gap between Step A and Step C.

Constraint: You must output your answer EXCLUSIVELY as a valid JSON object. Do not include any conversational text.
Use this exact schema:
{{
    "name": "Short Name of Step B",
    "instruction": "The clear instruction connecting A to C."
}}<end_of_turn>
<start_of_turn>user
[Step A] Name: Read Context | Instr: Review the provided text carefully.
[Step C] Name: Filter Noise | Instr: Discard statements that do not directly contribute to the question.<end_of_turn>
<start_of_turn>model
```json
{{
    "name": "Isolate Claims",
    "instruction": "Extract the specific factual claims from the text to prepare them for evaluation."
}}
```<end_of_turn>
<start_of_turn>user
[Step A] Name: Identify Entities | Instr: List all the people mentioned in the document.
[Step C] Name: Build Graph | Instr: Create a network graph detailing how the entities are connected.<end_of_turn>
<start_of_turn>model
```json
{{
    "name": "Determine Relationships",
    "instruction": "Analyze the text surrounding the identified people to define the exact nature of their relationships."
}}
```<end_of_turn>
<start_of_turn>user
[Step A] Name: Translate Text | Instr: Translate the paragraph into French.
[Step C] Name: Format Output | Instr: Return the final output enclosed in markdown bold tags.<end_of_turn>
<start_of_turn>model
```json
{{
    "name": "Review Translation",
    "instruction": "Check the French translation for grammatical accuracy and ensure it maintains a natural tone."
}}
```<end_of_turn>
<start_of_turn>user
[Step A] Name: {name1} | Instr: {inst1}
[Step C] Name: {name2} | Instr: {inst2}<end_of_turn>
<start_of_turn>model
"""
        try:
            response: str = await self.llm.generate_text(bridge_prompt, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
            
            # Robust JSON extraction: Find the first { and last }
            import json
            start_idx = response.find('{')
            end_idx = response.rfind('}') + 1
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                name = data.get("name", "").strip()
                instruction = data.get("instruction", "").strip()
                
                if len(instruction) >= 3:
                    embedding = self.llm.get_embedding(instruction)
                    return PromptNode(name, instruction, embedding=embedding, innovation_number=-1)
                    
            print(f"Bridge JSON Parsing Failed. Raw response: {response}")
            return None
            
        except Exception as e:
            print(f"Bridge Exception: {e}")
            return None
    
    async def _split_instructions(self, original_instruction: str, original_name: str) -> tuple[str, str, str, str]:
        split_prompt = f"""<start_of_turn>system
You are an expert cognitive architect. Your task is to split a complex instruction into two sequential, atomic sub-steps (Part 1: Preparation, Part 2: Execution).

Constraint: Output EXCLUSIVELY as a valid JSON array of two objects.
CRITICAL TERMINAL RULE: If the Original Instr contains a strict output format (e.g., "answer with exactly one word", "output only the category"), you MUST append that constraint verbatim to the Step 2 instruction.

Schema:
[
    {{"name": "Step 1 Name", "instruction": "Step 1 Instruction"}},
    {{"name": "Step 2 Name", "instruction": "Step 2 Instruction"}}
]<end_of_turn>
<start_of_turn>user
Original Name: Evaluate Sentiment
Original Instr: Analyze the review and express its sentiment using exactly one word.<end_of_turn>
<start_of_turn>model
```json
[
    {{"name": "Extract Keywords", "instruction": "Identify the key adjectives and emotive phrases in the review."}},
    {{"name": "Classify Sentiment", "instruction": "Based on the extracted keywords, express the sentiment using exactly one word."}}
]
```<end_of_turn>
<start_of_turn>user
Original Name: Process Invoice
Original Instr: Extract the total amount due from the invoice and convert it to USD.<end_of_turn>
<start_of_turn>model
```json
[
    {{"name": "Locate Total Amount", "instruction": "Scan the provided invoice text and extract the final total amount due, including its original currency."}},
    {{"name": "Convert Currency", "instruction": "Take the extracted total amount and calculate its equivalent value in USD."}}
]
```<end_of_turn>
<start_of_turn>user
Original Name: Categorize Support Ticket
Original Instr: Read the customer email and classify it into billing, technical, or general inquiries, outputting only the category name.<end_of_turn>
<start_of_turn>model
```json
[
    {{"name": "Analyze Email Content", "instruction": "Read the customer email to determine the primary issue or question being raised."}},
    {{"name": "Assign Category", "instruction": "Classify the identified issue into billing, technical, or general inquiries, outputting only the category name."}}
]
```<end_of_turn>
<start_of_turn>user
Original Name: {original_name}
Original Instr: {original_instruction}<end_of_turn>
<start_of_turn>model
"""
        
        try:
            response: str = await self.llm.generate_text(split_prompt, max_tokens=256, temperature=MUTATIONS_TEMPERATURE)
            
            import json
            start_idx = response.find('[')
            end_idx = response.rfind(']') + 1
            
            if start_idx != -1 and end_idx != 0:
                json_str = response[start_idx:end_idx]
                data = json.loads(json_str)
                
                if len(data) == 2:
                    return (
                        data[0]["name"].strip(), data[0]["instruction"].strip(),
                        data[1]["name"].strip(), data[1]["instruction"].strip()
                    )

            raise ValueError("Invalid JSON array length or format.")

            
        except Exception as e:
            # Safe Fallback
            prep_name = f"Prepare_{original_name.replace(' ', '_')}"
            prep_instruction = "Analyze the provided context and explicitly map out the entities involved before proceeding."
            return prep_name, prep_instruction, original_name, original_instruction
        