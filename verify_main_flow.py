
import sys
import os
import asyncio
import types
from unittest.mock import MagicMock

# --- 1. MOCKING UTILS.LLM BEFORE IMPORTING MAIN ---
# We need to mock Utils.LLM.LLM before main.py imports it and tries to load SentenceTransformer
sys.path.append(os.path.join(os.getcwd(), 'Edoardo'))

# Create a mock module for Utils.LLM
mock_llm_module = types.ModuleType("Utils.LLM")
sys.modules["Utils.LLM"] = mock_llm_module

class MockLLM:
    def __init__(self, base_url=None):
        print("MOCK LLM INITIALIZED")
    
    def get_embedding(self, text):
        return [0.1, 0.2, 0.3] # Dummy embedding (1D)
    
    async def generate_text(self, user_prompt, max_tokens=512, stop=None, temperature=0.2, primer=""):
        return "This is a mock response from the LLM."

mock_llm_module.LLM = MockLLM

# --- 2. IMPORTING MAIN ---
# Now we can safely import main. It will use our MockLLM
import main

# --- 3. MONKEYPATCHING CONFIGURATION ---
# Reduce scope for quick verification
main.MAX_GENERATIONS = 2
main.USE_REASONING = False # Keep it simple
main.TARGET_FITNESS = 100.0 # Don't stop early unless we hit this high value
main.MAX_TIME_SECONDS = 30 # Short timeout

# We also need to monkeypatch initialize_population because main.py hardcodes num_individuals=30
original_init_pop = main.initialize_population
async def mock_init_pop(*args, **kwargs):
    print("MOCK INIT POPULATION: Reducing size to 4")
    kwargs['num_individuals'] = 4
    return await original_init_pop(*args, **kwargs)

main.initialize_population = mock_init_pop

# --- 4. RUNNING THE EVOLUTION ---
async def run_verification():
    print("\n🚀 STARTING MOCK VERIFICATION of main.py 🚀")
    try:
        # We need to suppress the subprocess.Popen xdg-open call because it might fail in headless env or annoy user
        import subprocess
        subprocess.Popen = MagicMock()
        
        await main.run_evolution()
        print("\n✅ MAIN FLOW VERIFICATION SUCCESSFUL")
    except Exception as e:
        print(f"\n❌ MAIN FLOW VERIFICATION FAILED: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(run_verification())
