import random
import copy
from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Gene.connection import Connection
from Utils.utilities import _get_next_innovation_number
#from Utils.MarkDownLogger import md_logger
from Utils.LLM import LLM

MUTATIONS_TEMPERATURE = 0.8
MATURITY_THRESHOLD = 7


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
        Architectural mutations act as a self-correcting thermostat centered at N=7 nodes.
        Gene mutations shift from early Cognitive Structuring to late Linguistic Compression.
        """
        
        # --- 1. Global Rate Calculations --- (for a mature graph there is a 50% chance of not mutating at all)
        # Architectural Decay: Cools down global topology changes over generations
        p_arch = max(0.35, 0.75 ** (generation + 1))
        # Gene Mutation: Ensures roughly 1 mutation per child graph
        p_gene = max(0.15, 0.75 / max(1, node_count))

        # --- 2. Architectural Probabilities (Thermostat to N=7) ---
        
        # A. Node Balance (Sum = 0.50)
        if node_count <= 7:
            # Scales from 0.0 at N=1 to 0.25 at N=7
            t_node = (node_count - 1) / 6.0 if node_count > 1 else 0.0
            p_remove_node = 0.25 * t_node
        else:
            # Scales from 0.25 at N=7 to 0.50 at N=14 (capped at 14)
            t_node = min(1.0, (node_count - 7) / 7.0)
            p_remove_node = 0.25 + (0.25 * t_node)
            
        p_add_node = 0.50 - p_remove_node

        # B. Connection Balance (Sum = 0.50) based on DAG Density
        if node_count <= 1:
            density = 1.0 # Cannot add connections if only 1 node exists
        else:
            max_possible_connections = (node_count * (node_count - 1)) / 2.0
            density = min(1.0, connections_count / max_possible_connections)
            
        p_remove_conn = 0.50 * density
        p_add_conn = 0.50 - p_remove_conn

        # --- 3. Gene Probabilities (Structuring -> Compression) ---
        
        # Alpha scalar: 0.0 when N=1, 1.0 when N>=7
        alpha = min(1.0, (node_count - 1) / 6.0) if node_count > 1 else 0.0

        def lerp(start: float, end: float, t: float) -> float:
            return start + (end - start) * t

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
                MutType.GENE_SPLIT:            lerp(0.30, 0.05, alpha),
                MutType.GENE_INJECT_REASONING: lerp(0.25, 0.10, alpha),
                MutType.GENE_EXPAND:           lerp(0.15, 0.05, alpha),
                MutType.GENE_ADD_PERSONA:      lerp(0.10, 0.15, alpha),
                MutType.GENE_REFORMULATE:      lerp(0.20, 0.35, alpha),
                MutType.GENE_SIMPLIFY:         lerp(0.00, 0.30, alpha)
            }
        }
    def __init__(self, breeder_llm_client: LLM, default_config=None):
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
    
    def _get_ancestors(self, genome: AgentGenome, target_node_id: str) -> set[str]:
        """
        Returns a set of IDs of all nodes that can reach target_node_id.
        Uses Reverse-BFS.
        """
        # 1. Build Reverse Adjacency List (Map: Node -> Parents)
        # Optimization: We build this only for the search. 
        # In a larger system, you might cache this on the genome.
        parents_map = {nin: [] for nin in genome.nodes.keys()}
        for conn in genome.connections.values():
            if conn.enabled:
                if conn.out_node in parents_map.keys():
                    parents_map[conn.out_node].append(conn.in_node)
        
        # 2. Perform BFS backwards from target
        ancestors = set()
        stack = [target_node_id]
        
        while stack:
            current = stack.pop()
            # If we haven't visited this node yet (it's a new ancestor)
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
        connection = random.choice(list(genome.connections.values()))
        if not connection: return

        # get name and instructions of the in_node and out_node
        in_node = genome.nodes[connection.in_node]
        out_node = genome.nodes[connection.out_node]

        # create a new node and insert it between in_node and out_node of the chosen connection
        new_node: PromptNode = await self._generate_new_node(in_node.name, in_node.instruction, out_node.name, out_node.instruction)
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
        
        for conn in genome.connections.values():
            if not conn.enabled: continue
            
            if conn.out_node == node_innovation_number:
                incoming_node_innovation_numbers.append(conn.in_node)
                conn.enabled = False
            elif conn.in_node == node_innovation_number:
                outgoing_node_innovation_numbers.append(conn.out_node)
                conn.enabled = False
        
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
        genome.nodes.pop(node_innovation_number, None)
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
        active_connections = [c for c in genome.connections.values() if c.enabled]
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
            #print(f"Global: Removing connection {target_conn.in_node}... -> {target_conn.out_node}...")
            target_conn.enabled = False
            #md_logger.log_event(f"""Removed connection from '{genome.nodes[target_conn.in_node].name}' to '{genome.nodes[target_conn.out_node].name}'""")
        else:
            print("Global: No removable connections found (all are critical bridges).")

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
                #md_logger.log_event(f"Applied mutation {mutation_type}")

        # 5. Gene Level Mutations (Per Node Check)
        await self._apply_gene_mutations(mutated_genome, p_mutate_node, gene_cdf)
        #md_logger.log_event(f"Applied mutations to {len(mutated_genome.nodes)} nodes")

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
                
                mut_type = self._pick_from_cdf(gene_cdf)

                if mut_type == MutType.GENE_SPLIT:
                    await self._handle_split(genome, node)
                elif mut_type == MutType.GENE_EXPAND:
                    await self._handle_content_mutation(genome.nodes[node], "expand")
                elif mut_type == MutType.GENE_SIMPLIFY:
                    await self._handle_content_mutation(genome.nodes[node], "simplify")
                elif mut_type == MutType.GENE_REFORMULATE:
                    await self._handle_content_mutation(genome.nodes[node], "reformulate")

    # --- Specific Mutation Handlers ---

    async def _handle_split(self, genome: AgentGenome, target_node: str):
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
        
        new_gene = genome.nodes[target_node].copy()  # Copy other attributes like type, embedding, etc.
        new_gene.name = name2
        new_gene.instruction = prompt2
        new_gene.embedding = self.llm.get_embedding(prompt2)
        new_gene.innovation_number = _get_next_innovation_number()

        genome.add_node(new_gene)

        # 3. Manage Output Connections
        # We need to find all connections LEAVING the target_node and move them to B.
        conns_to_disable: list[Connection] = []
        conns_to_create: list[str] = []

        for conn in genome.connections.values():
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
        genome.add_connection(target_node, new_gene.innovation_number)

        # C. Create the new outgoing connections from B
        for out_innovation_number in conns_to_create:
            genome.add_connection(new_gene.innovation_number, out_innovation_number)

    async def _handle_content_mutation(self, node: PromptNode, style: str):
        # Calculate a safe max_token limit based on input length
        # (Approx 1 token ~= 4 chars). We allow +50% growth max.
        current_len = len(node.instruction)
        safe_max_tokens = int((current_len / 3) * 1.5) + 64

        #print(f"Applying {style} mutation to node: {node.name}\nOriginal: {node.instruction}")

        expand_prompt = f"""Task: Expansion.
Rewrite the instruction to be more detailed by breaking it down into specific, actionable logical steps. 
Constraint: You must preserve only the exact original objective. Do not add new goals.

### Examples

Original: Identify the connection between the two subjects.
Detailed Version: First, locate the two target subjects in the provided text. Second, trace any intermediary links or shared attributes between them. Finally, synthesize these links to deduce their exact connection.

Original: Think step-by-step to find the answer.
Detailed Version: Break down the problem into individual logical components. Analyze each component sequentially, ensuring each deduction is strictly grounded in the provided facts before reaching the final conclusion.

Original: Filter the relevant information.
Detailed Version: Review the provided context and actively discard any extraneous details. Isolate only the specific facts that directly contribute to solving the core objective.

### Current Task
Original: {node.instruction}
"""

        expand_primer = "Detailed Version:"

        simplify_prompt = f"""Task: Simplification.
Compress the instruction into a single, concise high-level cognitive goal. Remove verbose step-by-step details.
Constraint: You must preserve the exact underlying goal and any strict output formats. Do not change *what* needs to be done, only simplify the description of *how* to do it.

### Examples

Original: First, locate the two target subjects in the provided text. Second, trace any intermediary links or shared attributes between them. Finally, synthesize these links to deduce their exact connection.
Simplified Version: Identify the connection between the subjects.

Original: Break down the problem into individual logical components. Analyze each component sequentially, ensuring each deduction is strictly grounded in the provided facts before reaching the final conclusion.
Simplified Version: Think step-by-step to reach a logical conclusion.

Original: Review the provided context and actively discard any extraneous details. Isolate only the specific facts that directly contribute to solving the core objective.
Simplified Version: Extract only the relevant facts.

### Current Task
Original: {node.instruction}
"""

        simplify_primer = "Simplified Version:"

        reformulate_prompt = f"""Task: Stylistic Paraphrasing.
Rewrite the instruction using different vocabulary and sentence structure while preserving the exact original cognitive goal. Do not add or remove logical steps.

Constraint: You must strictly preserve any specific output formatting rules (e.g., "State only the one word"). Change the phrasing of the thought process, but NEVER change the final output requirement.

Style Directive: Before writing, internally select and apply one of the following distinct cognitive styles:
1. Inquisitive (Framing the task as a direct question to the AI)
2. Concise & Direct (Minimalist, stripped of all filler words)
3. Academic & Formal (Highly precise, analytical language)
4. Imperative (Strong, commanding action verbs)

### Examples

(Inquisitive)
Original: State only the single kinship word that describes the relationship.
Reformulated Version: What specific kinship term defines their connection? Output only that single word.

(Concise & Direct)
Original: Think step-by-step to reach a logical conclusion.
Reformulated Version: Systematically derive the logical conclusion.

(Academic & Formal)
Original: Identify the connection between the subjects.
Reformulated Version: Deduce the exact relational link connecting the specified entities.

(Imperative)
Original: Read the context carefully and extract the right answer.
Reformulated Version: Isolate the correct answer directly from the provided context.

### Current Task
Original: {node.instruction}
"""

        reformulate_primer = "Reformulated Version:"

        persona_prompt = f"""Task: Persona Injection.
Rewrite the instruction by commanding the AI to adopt a highly specific, expert cognitive persona. If a persona is already implied, change it to a different one.

Constraint: You must preserve the exact underlying goal and any strict output formatting rules. Wrap the original logic in the persona's perspective, but NEVER change the required format of the final answer.

Persona Directive: Before writing, internally select and apply one of the following distinct expert archetypes:
1. The Master Detective (Focuses on clues, evidence tracing, and hidden links)
2. The Rigorous Mathematician (Focuses on axioms, variables, and absolute proof)
3. The Skeptical Auditor (Focuses on verifying facts and actively discarding assumptions)
4. The Systems Engineer (Focuses on structural relationships and node mappings)

### Examples

(The Master Detective)
Original: Identify the connection between the subjects.
Mutated Version: Act as a world-class detective. Scour the provided text for clues and trace the evidence to deduce the exact connection between the subjects.

(The Rigorous Mathematician)
Original: Think step-by-step to reach a logical conclusion.
Mutated Version: You are a brilliant mathematician. Construct a rigorous, step-by-step proof derived solely from the provided axioms to reach an undeniable logical conclusion.

(The Skeptical Auditor)
Original: Extract only the relevant facts.
Mutated Version: As a strict auditor, review the context with intense skepticism. Discard all extraneous noise and isolate only the verified facts required for the objective.

(The Systems Engineer)
Original: State only the single kinship word that describes the relationship.
Mutated Version: You are an expert systems engineer mapping a relational graph. What specific terminal node (kinship term) defines their connection? Output only that single word.

### Current Task
Original: {node.instruction}
"""

        persona_primer = "Mutated Version:"

        reasoning_prompt = f"""Task: Reasoning Injection.
Enhance the instruction by explicitly commanding the AI to apply a specific cognitive reasoning framework before generating its final answer. If a reasoning style is already implied, add a new one to further deepen the thought process.

Constraint: You must preserve the exact underlying goal and any strict output formatting rules. You may add instructions on *how* to think before answering, but NEVER change the required format of the final answer.

Reasoning Directive: Before writing, internally select and apply one of the following distinct reasoning frameworks:
1. Chain of Thought (Force a sequential, step-by-step logical progression)
2. Working Backwards (Start from the target objective and trace dependencies backward)
3. Falsification (Explicitly rule out incorrect options before settling on the true answer)
4. Sub-goal Decomposition (Break the main problem into smaller, independent questions to answer sequentially)

### Examples

(Chain of Thought)
Original: Deduce the exact relational link connecting the specified entities.
Mutated Version: Deduce the exact relational link connecting the specified entities. Before answering, carefully map out the logical progression step-by-step to ensure absolute accuracy.

(Working Backwards)
Original: Identify the connection between the subjects.
Mutated Version: Identify the connection between the subjects. Begin by looking at the final subject and logically work backward through their direct dependencies until you reach the starting subject.

(Falsification)
Original: State only the single kinship word that describes the relationship.
Mutated Version: First, internally review all possible relationships and explicitly rule out the logically impossible ones. Then, state only the single kinship word that describes the true relationship.

(Sub-goal Decomposition)
Original: Extract only the relevant facts.
Mutated Version: First, divide the text into independent informational chunks. Evaluate the relevance of each chunk individually. Finally, extract only the relevant facts required for the objective.

### Current Task
Original: {node.instruction}
"""

        reasoning_primer = "Mutated Version:"

        # Select the prompt
        if style == "expand":
            prompt, primer = expand_prompt, expand_primer
        elif style == "simplify":
            prompt, primer = simplify_prompt, simplify_primer
        elif style == "persona":
            prompt, primer = persona_prompt, persona_primer
        elif style == "reasoning":
            prompt, primer = reasoning_prompt, reasoning_primer
        else:
            prompt, primer = reformulate_prompt, reformulate_primer

        # CRITICAL FIX: Temperature set to 0.8 to force genetic diversity
        response: str = await self.llm.generate_text(prompt, max_tokens=safe_max_tokens, temperature=MUTATIONS_TEMPERATURE, primer=primer)
        
        if response and len(response.strip()) > 5:
            node.instruction = response.strip()
            node.embedding = self.llm.get_embedding(response)
        else:
            print(f"Failed to generate new instruction for {node.name}. Retaining original.")
            
    async def _generate_new_node(self, name1: str, inst1: str, name2: str, inst2: str) -> PromptNode:
        """
        Asks for a bridging cognitive step.
        Returns: A new PromptNode object connecting Step A and Step C.
        """
        bridge_prompt = f"""System Directive: You are an abstract cognitive routing engine. Your goal is to construct generalized reasoning pathways.

Task: Logical Bridge Insertion.
Create a missing intermediate cognitive step that logically bridges the gap between Step A and Step C. 

Constraint: You must output exactly ONE intermediate step using the strict format `Name: <name> | Instr: <instruction>`. Do not add any conversational filler, and do not create more than one step.

### Examples

[Step A] Name: Extract Variables | Instr: Identify the core subjects and conditions stated in the premise.
[Step C] Name: Final Deduction | Instr: Synthesize the findings into a single verifiable conclusion.
[Step B] Name: Formulate Dependencies | Instr: Map out how the extracted variables interact and logically depend on one another.

[Step A] Name: Define Objective | Instr: Clarify the exact question that needs to be answered.
[Step C] Name: Execute Solution | Instr: Calculate or derive the final answer based on the established framework.
[Step B] Name: Devise Strategy | Instr: Formulate a step-by-step logical plan to move from the initial objective to the final solution.

[Step A] Name: Read Context | Instr: Review the provided text carefully.
[Step C] Name: Filter Noise | Instr: Discard all statements that do not directly contribute to the target question.
[Step B] Name: Isolate Facts | Instr: Extract the specific factual claims from the context so they can be evaluated.

### Current Task
[Step A] Name: {name1} | Instr: {inst1}
[Step C] Name: {name2} | Instr: {inst2}
"""

        bridge_primer = "[Step B] Name:"
        
        response: str = await self.llm.generate_text(bridge_prompt, max_tokens=128, temperature=MUTATIONS_TEMPERATURE, primer=bridge_primer)
        
        # Robust Parsing
        try:
            full_text = "Name:" + response 
            name = full_text.split("Name:")[1].split("| Instr:")[0].strip()
            instruction = full_text.split("Instr:")[1].strip()
            
            embedding = self.llm.get_embedding(instruction)
            return PromptNode(name, instruction, embedding=embedding, innovation_number=_get_next_innovation_number())
            
        except Exception as e:
            # print(f"Bridge Parsing Failed: {e}. Using cognitive fallback.")
            
            # Use a domain-agnostic cognitive instruction for the fallback to maintain fitness
            fallback_instruction = "Review the deductions made so far and explicitly map out the logical dependencies required for the next step."
            fallback_embedding = self.llm.get_embedding(fallback_instruction)
            
            return PromptNode("Logical_Bridge", fallback_instruction, embedding=fallback_embedding, innovation_number=_get_next_innovation_number())
    
    async def _split_instructions(self, original_instruction: str, original_name: str) -> tuple[str, str, str, str]:
        """
        Forces strict splitting format into two atomic cognitive steps.
        Returns: (name1, prompt1, name2, prompt2)
        """
        prompt = f"""System Directive: You are an abstract cognitive routing engine. Your goal is to construct generalized reasoning pathways.

Task: Sequential Logic Split.
Split the Original Instruction into two sequential, atomic cognitive steps (Part 1 then Part 2) that together achieve the exact same goal.

Constraint: You must output exactly TWO steps using the strict format `1. Name: <name> | Instr: <instruction>`. 
CRITICAL: If the Original Instruction contains a strict output format or grammatical trap (e.g., asking for a single word), you MUST place that exact formatting constraint in the instruction for Step 2. Do not let Step 1 generate the final answer.

### Examples

Original Name: Synthesize and Answer
Original Instr: Trace the lineage from Person A to B and output the exact kinship term.
1. Name: Trace Lineage | Instr: Map the step-by-step relational lineage from Person A to Person B based on the text.
2. Name: Output Kinship | Instr: Based on the traced lineage, state only the exact kinship term that defines their connection.

Original Name: Isolate and Deduce
Original Instr: Discard irrelevant facts and logically deduce the hidden connection.
1. Name: Isolate Facts | Instr: Review the text and explicitly discard all irrelevant statements, keeping only the core facts.
2. Name: Deduce Connection | Instr: Use the isolated facts to logically deduce the final hidden connection between the subjects.

### Current Task
Original Name: {original_name}
Original Instr: {original_instruction}
"""
        primer = "1. Name:"
        
        response: str = await self.llm.generate_text(prompt, temperature=MUTATIONS_TEMPERATURE, primer=primer)
        
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
            prep_instruction = "Analyze the provided context and mentally map out the entities involved before proceeding to the final task."
            
            return prep_name, prep_instruction, original_name, original_instruction