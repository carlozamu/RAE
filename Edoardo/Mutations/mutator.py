import random
import copy
from Edoardo.Genome.agent_genome import AgentGenome
from Edoardo.Gene.gene import PromptNode
from Edoardo.Gene.connection import Connection
from Utils.LLM import LLM

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
        parents_map = {nid: [] for nid in genome.nodes}
        for conn in genome.connections.values():
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
        genome.add_connection(connection.in_node, new_node.id)
        genome.add_connection(new_node.id, connection.out_node)

    def _handle_remove_node(self, genome: AgentGenome):
        # choose a random node to remove
        node_id: str = random.choice(list(genome.nodes.keys()))
        if node_id == genome.start_node_id or node_id == genome.end_node_id:
            return # do not remove start or end nodes
        
        incoming_node_ids = []
        outgoing_node_ids = []
        
        for conn in genome.connections.values():
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
            for conn in genome.connections.values():
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
            print(f"Global: Removing connection {target_conn.in_node[:8]}... -> {target_conn.out_node[:8]}...")
            target_conn.enabled = False
        else:
            print("Global: No removable connections found (all are critical bridges).")

    # ------- Main Mutation Logic -------
    async def mutate(self, genome: AgentGenome, runtime_config=None) -> AgentGenome:
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
        mutated_genome = copy.deepcopy(genome)

        # 4. Global Architectural Mutation (Single Event)
        if random.random() < p_arch_event:
            mutation_type = self._pick_from_cdf(arch_cdf)
            if mutation_type:
                await self._apply_global_mutation(mutated_genome, mutation_type)

        # 5. Gene Level Mutations (Per Node Check)
        await self._apply_gene_mutations(mutated_genome, p_mutate_node, gene_cdf)

        return mutated_genome

    # --- Internal Logic ---
    async def _apply_global_mutation(self, genome: AgentGenome, mutation_type: MutType):
        """Dispatches architectural mutations."""
        print(f"Applying Global Mutation: {mutation_type}")
        
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
        print(f"Splitting node: {genome.nodes[target_node].name}")
        
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
        genome.add_connection(target_node, new_gene.id)

        # C. Create the new outgoing connections from B
        for out_id in conns_to_create:
            genome.add_connection(new_gene.id, out_id)

    async def _handle_content_mutation(self, node: PromptNode, style: str):
        # Calculate a safe max_token limit based on input length
        # (Approx 1 token ~= 4 chars). We allow +50% growth max.
        current_len = len(node.instruction)
        safe_max_tokens = int((current_len / 3) * 1.5) + 64

        print(f"Applying {style} mutation to node: {node.name}\nOriginal: {node.instruction}")

        expand_prompt = f"""Task: Expansion.
Rewrite the instruction to be more detailed by breaking it down into specific, actionable steps.

### Examples

Original: Read the user input csv file.
Detailed Version: Open the csv file at the specified path, then parse the rows to extract column headers and values.

Original: Extract the 'price' column and calculate mean.
Detailed Version: Select the 'price' column from the dataframe, convert values to numeric, and compute the arithmetic average.

Original: Decompose the problem.
Detailed Version: Analyze the main objective, break it down into a series of atomic sub-tasks, and solve them sequentially.

### Current Task
Original: {node.instruction}
"""
        expand_primer = "Detailed Version:"

        simplify_prompt = f"""Task: Simplification.
Compress the instruction into a single, concise high-level goal. Remove implementation details.

### Examples

Original: Open the csv file at the specified path, then parse the rows to extract column headers and values.
Simplified Version: Read the CSV file.

Original: Select the 'price' column from the dataframe, convert values to numeric, and compute the arithmetic average.
Simplified Version: Calculate the average price.

Original: Analyze the main objective, break it down into a series of atomic sub-tasks, and solve them sequentially.
Simplified Version: Decompose the problem.

### Current Task
Original: {node.instruction}
"""
        simplify_primer = "Simplified Version:"

        reformulate_prompt = f"""Task: Paraphrasing.
Rewrite the instruction using different vocabulary and sentence structure while preserving the exact original meaning. Do not add or remove steps.

### Examples

Original: Calculate the sum of the array elements.
Reformulated Version: Compute the total value of all items in the list.

Original: If the file does not exist, create it.
Reformulated Version: Generate a new file if one is not found at the location.

Original: Sort the data by date in descending order.
Reformulated Version: Order the dataset from newest to oldest based on the date.

### Current Task
Original: {node.instruction}
"""
        reformulate_primer = "Reformulated Version:"

        # Select the prompt
        if style == "expand":
            prompt = expand_prompt
            primer = expand_primer
        elif style == "simplify":
            prompt = simplify_prompt
            primer = simplify_primer
        else:
            prompt = reformulate_prompt
            primer = reformulate_primer

        response: str = await self.llm.generate_text(prompt, max_tokens=safe_max_tokens, temperature=0.2, primer=primer)
        
        if response and len(response.strip()) > 5:
            print(f"New {style} Instruction: \n{response}\n\n")
            node.instruction = response.strip()
            node.embedding = self.llm.get_embedding(response)
        else:
            print(f"Failed to generate new instruction for {node.name}")

    async def _generate_new_node(self, name1: str, inst1: str, name2: str, inst2: str) -> PromptNode:
        """
        Asks for a bridging step.
        Returns: (name, instruction)
        """
        print(f"Applying bridge mutation to nodes: {name1} -> {name2},\nOriginal Instructions: \n1:{inst1}\n2:{inst2}")
        prompt = f"""Task: Create a missing intermediate step that logically connects Step A and Step C.
Format: Name: <name> | Instr: <instruction>

### Examples

[Step A] Name: Get Text | Instr: Extract the raw text from the input source.
[Step C] Name: Summarize | Instr: Write a one-sentence summary of the content.
[Step B] Name: Clean Text | Instr: Remove special characters and normalize the whitespace in the text.

[Step A] Name: Plan Course | Instr: Outline the high-level goals for a history course.
[Step C] Name: Create Exam | Instr: Write a final exam testing the students' knowledge.
[Step B] Name: Create Lessons | Instr: Develop detailed lesson plans and lectures covering the course goals.

[Step A] Name: Clean Data | Instr: Remove duplicates and outliers from the dataset.
[Step C] Name: Create Report | Instr: Generate a comprehensive report summarizing the findings.
[Step B] Name: Analyze Data | Instr: Analyze the dataset for patterns and insights.

### Current Task
[Step A] Name: {name1} | Instr: {inst1}
[Step C] Name: {name2} | Instr: {inst2}
"""
        primer = "[Step B] Name:"
        # We prime the model with "Name:" so it completes the rest
        response: str = await self.llm.generate_text(prompt, max_tokens=128, temperature=0.2, primer=primer)
        print(f"New Node:\nName:{response}\n\n")
        
        # Robust Parsing
        try:
            # We prepend "Name:" because the model completes it
            full_text = "Name:" + response 
            name = full_text.split("Name:")[1].split("| Instr:")[0].strip()
            print(f"Name: {name}")
            instruction = full_text.split("Instr:")[1].strip()
            print(f"Instruction: {instruction}")
            embedding = self.llm.get_embedding(instruction)
            return PromptNode(name, instruction, embedding=embedding)
        except:
            return PromptNode("Bridge_Step", "Process the data from the previous step and prepare it for the next.", embedding=self.llm.get_embedding(instruction))

    async def _split_instructions(self, original_instruction: str, original_name: str) -> tuple[str, str, str, str]:
        """
        Forces strict splitting format.

        Returns: (name1, prompt1, name2, prompt2)
        """
        print(f"\nApplying split mutation to instruction:\n {original_instruction}\n")
        prompt = f"""Task: Split the Original Instruction into two sequential, atomic steps (Part 1 then Part 2).
Format:
1. Name: <name> | Instr: <instruction>
2. Name: <name> | Instr: <instruction>

### Examples

Original Name: Mean Computation
Original Instr: Extract the 'price' column and calculate the mean value.
1. Name: Extract Column | Instr: Select the 'price' column from the dataset.
2. Name: Calculate Mean | Instr: Compute the arithmetic average of the selected values.

Original Name: Summarize
Original Instr: Read the text and produce a concise summary.
1. Name: Read Text | Instr: Parse the input text to understand the main points.
2. Name: Write Summary | Instr: Generate a concise summary based on the parsed text.

### Current Task
Original Name: {original_name}
Original Instr: {original_instruction}
"""
        primer = "1. Name:"
        
        response: str = await self.llm.generate_text(prompt, max_tokens=512, temperature=0.2, primer=primer)
        print(f"Split Instructions:\n{response}\n\n")
        
        try:
            # Parsing logic for "1. Name: ... | Instr: ..."
            full_text = "1. Name:" + response
            lines = full_text.splitlines()
            
            # Extract Line 1
            part1 = lines[0].split("| Instr:")
            n1 = part1[0].replace("1. Name:", "").strip()
            i1 = part1[1].strip()
            
            # Extract Line 2 (Find the line starting with "2.")
            line2 = next(l for l in lines if l.strip().startswith("2."))
            part2 = line2.split("| Instr:")
            n2 = part2[0].replace("2. Name:", "").strip()
            i2 = part2[1].strip()

            print(f"Name 1: {n1}")
            print(f"Instruction 1: {i1}")
            print(f"Name 2: {n2}")
            print(f"Instruction 2: {i2}")   
            
            return n1, i1, n2, i2
        except:
            return original_name, original_instruction, "Refine results", "Refine and improve results"
