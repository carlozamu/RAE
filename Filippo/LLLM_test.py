import asyncio
import random
from LLM import LLM
from AgentGenome import AgentGenome, PromptNode
from Mutator import Mutator, MutType

async def test_mutator_logic():
    print("\n=== 🧬 STARTING MUTATOR INTEGRATION TEST ===\n")
    
    # 1. Setup
    print("[1] Connecting to Breeder LLM...")
    try:
        llm = LLM() 
    except Exception as e:
        print(f"Server offline? {e}")
        return

    mutator = Mutator(llm)
    
    # 2. Seed Genome
    print("[2] Creating Seed Genome...")
    genome = AgentGenome()
    # IDs: 'start', 'mid', 'end' for easier debugging logic
    start = PromptNode("Start", "Read the user input csv file.", llm.get_embedding("Read input"), node_id="start")
    genome.start_node_id = start.id
    
    mid = PromptNode("Analyze", "Extract the column 'price' and calculate mean.", llm.get_embedding("Calculate mean"), node_id="mid")
    
    end = PromptNode("Output", "Format the result as JSON.", llm.get_embedding("Format JSON"), node_id="end")
    genome.end_node_id = end.id
    
    genome.add_node(start)
    genome.add_node(mid)
    genome.add_node(end)
    
    genome.add_connection("start", "mid")
    genome.add_connection("mid", "end")
    
    print(f"   Initial Chain: {[n.name for n in genome.get_linear_chain()]}")

    # --- HELPER TO RESET GENOME ---
    def get_fresh_genome():
        return genome.copy()

    # 3. Test EXPAND (Specific Node)
    print("\n[3] Testing GENE_EXPAND (Targeting 'Analyze')...")
    # We can't target a specific node easily with probability 1.0 on all.
    # But we can verify it happened.
    config = {
        "p_architectural_event": 0.0,
        "p_mutate_node": 1.0, # All nodes will be hit
        "gene_probs": { MutType.GENE_EXPAND: 1.0 }
    }
    g_expand = await mutator.mutate(get_fresh_genome(), runtime_config=config)
    
    # Check 'mid' node
    new_instr = g_expand.nodes["mid"].instruction
    print(f"   Old: {mid.instruction}")
    print(f"   New: {new_instr[:100]}...")
    if len(new_instr) > len(mid.instruction):
        print("   ✅ SUCCESS: Expanded.")
    else:
        print("   ⚠️ WARNING: No expansion.")

    # 4. Test SIMPLIFY
    print("\n[4] Testing GENE_SIMPLIFY...")
    # First, let's create a genome with a LONG instruction to simplify
    g_long = get_fresh_genome()
    g_long.nodes["mid"].instruction = "Open the file, read the lines, iterate through them, find the column named price, sum them up, divide by count."
    
    config = {
        "p_architectural_event": 0.0,
        "p_mutate_node": 1.0,
        "gene_probs": { MutType.GENE_SIMPLIFY: 1.0 }
    }
    g_simple = await mutator.mutate(g_long, runtime_config=config)
    
    new_instr = g_simple.nodes["mid"].instruction
    print(f"   Old: {g_long.nodes['mid'].instruction}")
    print(f"   New: {new_instr}")
    if len(new_instr) < len(g_long.nodes["mid"].instruction):
        print("   ✅ SUCCESS: Simplified.")
    else:
        print("   ⚠️ WARNING: No simplification.")

    # 5. Test REFORMULATE
    print("\n[5] Testing GENE_REFORMULATE...")
    config = {
        "p_architectural_event": 0.0,
        "p_mutate_node": 1.0,
        "gene_probs": { MutType.GENE_REFORMULATE: 1.0 }
    }
    g_ref = await mutator.mutate(get_fresh_genome(), runtime_config=config)
    
    print(f"   Old: {mid.instruction}")
    print(f"   New: {g_ref.nodes['mid'].instruction}")
    if g_ref.nodes["mid"].instruction != mid.instruction:
        print("   ✅ SUCCESS: Reformulated.")
    else:
        print("   ❌ FAILURE: Text unchanged.")

    # 6. Test ARCH_ADD_NODE
    print("\n[6] Testing ARCH_ADD_NODE...")
    config = {
        "p_architectural_event": 1.0,
        "arch_probs": { MutType.ARCH_ADD_NODE: 1.0 },
        "p_mutate_node": 0.0
    }
    g_add = await mutator.mutate(get_fresh_genome(), runtime_config=config)
    chain = g_add.get_linear_chain()
    print(f"   New Chain: {[n.name for n in chain]}")
    if len(chain) == 4:
        print("   ✅ SUCCESS: Node added.")
    else:
        print("   ❌ FAILURE: Chain length same.")

    # 7. Test GENE_SPLIT
    print("\n[7] Testing GENE_SPLIT...")
    # This is the tricky one.
    config = {
        "p_architectural_event": 0.0,
        "p_mutate_node": 1.0, # Will try to split ALL nodes
        "gene_probs": { MutType.GENE_SPLIT: 1.0 }
    }
    g_split = await mutator.mutate(get_fresh_genome(), runtime_config=config)
    chain = g_split.get_linear_chain()
    print(f"   New Chain: {[n.name for n in chain]}")
    
    # If 3 nodes -> split all 3 -> we expect roughly 6 nodes (or 5 if start/end handled differently)
    if len(chain) > 3:
        print(f"   ✅ SUCCESS: Chain grew to {len(chain)} nodes.")
    else:
        print("   ❌ FAILURE: Chain length same.")

if __name__ == "__main__":
    asyncio.run(test_mutator_logic())