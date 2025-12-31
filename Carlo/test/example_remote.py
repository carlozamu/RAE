"""
Example: run a phenotype using Filippo's remote LLM server.
Requires the LLM server to be running (e.g., vLLM on localhost:8000).
"""
import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Filippo.AgentGenome import AgentGenome
from Filippo.Gene import PromptNode
from Filippo.LLM import LLM
from Carlo.model.llm_remote import LLMClientRemote
from Carlo.phenotype import Phenotype


def build_sample_genome() -> AgentGenome:
    genome = AgentGenome()

    node1 = PromptNode(
        name="Summarize",
        instruction="Summarize the input in one concise sentence."
    )
    node2 = PromptNode(
        name="Answer",
        instruction="Give a direct answer in one short sentence."
    )

    genome.add_node(node1)
    genome.add_node(node2)
    genome.start_node_id = node1.id
    genome.add_connection(node1.id, node2.id)
    return genome


def main():
    # Crea singleton LLM (connessione al server)
    llm_server = LLM(base_url="http://localhost:8000/v1")
    
    # Wrappa in adapter compatibile con LLMClient
    llm_client = LLMClientRemote(llm_server)
    
    # Costruisce phenotype
    genome = build_sample_genome()
    phenotype = Phenotype(genome, llm_client)
    
    print(f"Phenotype creato - età: {phenotype.age}, alive: {phenotype.alive}")
    print(f"Min age per eliminazione: {phenotype.min_age}")
    print(f"Può essere eliminato? {phenotype.can_be_eliminated()}")
    print()

    # Esegue
    outputs = phenotype.run(
        initial_input="Explain what NEAT does in neural network evolution.",
        answer_only=True
    )
    
    print("=== Outputs ===")
    for idx, out in enumerate(outputs, start=1):
        print(f"Node {idx}: {out}")
    
    print()
    print(f"Età dopo run: {phenotype.age}")
    print(f"Può essere eliminato? {phenotype.can_be_eliminated()}")
    
    print()
    print("=== Call Log (debug) ===")
    for entry in phenotype.call_log:
        print(f"[{entry['node_name']}] prompt length: {len(entry['prompt'])}, response length: {len(entry['response'])}")


if __name__ == "__main__":
    main()
