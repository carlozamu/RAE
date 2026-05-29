import asyncio
import random
import re
# Make sure to import your LLM class here
from llm_module import LLM 

# A block of generic text to simulate "Reasoning/Intermediate output" from parent nodes.
# This block is roughly 100 words (approx. 130 tokens).
GENERIC_REASONING_BLOCK = (
    "In analyzing the parameters of the current subsystem, we must consider the logical constraints "
    "defined by the overarching architecture. The state variables indicate a nominal execution path, "
    "with secondary metrics falling within acceptable operational bounds. We iterate over the contextual "
    "heuristics to ensure no edge cases trigger a cascaded failure. Furthermore, the cross-referencing "
    "of topological data reveals consistent alignment with the expected baseline. Moving forward, the "
    "system should maintain this trajectory, continually polling the active nodes for state updates "
    "and ensuring that memory buffers remain optimal during the synthesis phase. Next execution step approved."
)

async def run_context_test():
    llm = LLM(model_name="google/gemma-3-1b-it", base_url="http://localhost:8000") # Adjust URL/model if needed
    
    # We will test simulated graph depths (number of intermediate nodes)
    # 0 nodes = just the start and end (baseline)
    # 5 nodes = ~500 words of noise
    # up to 60 nodes = ~6000 words of noise (pushing SLM limits)
    depths_to_test = [0, 2, 4, 6, 8, 10, 15, 20]
    trials_per_depth = 3 # Run 3 times per depth to average out random hallucinations
    
    secret_key = "OMEGA-9942-X"
    
    print(f"{'Graph Depth':<15} | {'Approx Words':<15} | {'Success Rate':<15} | {'Sample Output'}")
    print("-" * 70)

    for depth in depths_to_test:
        success_count = 0
        last_output = ""
        
        for trial in range(trials_per_depth):
            # 1. Build the Gemma Chat History
            prompt = ""
            
            # --- START NODE (The Needle) ---
            prompt += "<start_of_turn>user\n"
            prompt += f"System Initialization: Register the following critical configuration key into your memory: {secret_key}. Acknowledge receipt.\n"
            prompt += "<end_of_turn>\n"
            
            prompt += "<start_of_turn>model\n"
            prompt += f"Acknowledged. The critical configuration key is {secret_key}. I will retain this for the final synthesis.\n"
            prompt += "<end_of_turn>\n"
            
            # --- INTERMEDIATE NODES (The Haystack) ---
            for i in range(depth):
                prompt += "<start_of_turn>user\n"
                prompt += f"Execute intermediate reasoning step {i+1} based on current state.\n"
                prompt += "<end_of_turn>\n"
                
                prompt += "<start_of_turn>model\n"
                # Add slight random variations to the noise so the LLM doesn't optimize it away
                noise = GENERIC_REASONING_BLOCK.replace("subsystem", f"subsystem {random.randint(1,100)}")
                prompt += f"Step {i+1} Output: {noise}\n"
                prompt += "<end_of_turn>\n"

            # --- END NODE (The Test) ---
            prompt += "<start_of_turn>user\n"
            prompt += "Final Task: Based on the very first initialization step of this execution thread, what is the critical configuration key? Output ONLY the key, with no other text.\n"
            prompt += "<end_of_turn>\n"
            
            # Append the final model start tag so it knows to answer
            prompt += "<start_of_turn>model\n"

            # 2. Call the LLM
            response = await llm.generate_text(
                user_prompt=prompt, 
                temperature=0.0, # 0.0 for strict factual recall
                max_tokens=20 
            )
            
            last_output = response.strip()
            
            # 3. Evaluate Success
            if secret_key in last_output:
                success_count += 1
                
        # Calculate stats
        approx_words = depth * 110 # ~100 words for model, ~10 for user per depth
        success_rate = (success_count / trials_per_depth) * 100
        
        # Clean up output for console
        display_out = last_output.replace('\n', ' ')[:30] + "..." if len(last_output) > 30 else last_output
        print(f"{depth:<15} | {approx_words:<15} | {success_rate:>13.1f}% | {display_out}")

if __name__ == "__main__":
    asyncio.run(run_context_test())
