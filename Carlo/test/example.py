"""
Minimal end-to-end example using the existing genome classes and the stub LLM.
"""
import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Filippo.AgentGenome import AgentGenome
from Filippo.Gene import PromptNode
from Carlo.model.llm_client import LLMClient
from Carlo.phenotype import Phenotype


def build_sample_genome() -> AgentGenome:
    genome = AgentGenome()

    node1 = PromptNode(name="Summarize", instruction="Summarize the input in one sentence.")
    node2 = PromptNode(name="Translate", instruction="Translate the previous summary to Italian.")

    genome.add_node(node1)
    genome.add_node(node2)

    genome.start_node_id = node1.id
    genome.add_connection(node1.id, node2.id)

    return genome


def main():
    genome = build_sample_genome()
    llm = LLMClient()
    phenotype = Phenotype(genome, llm)

    outputs = phenotype.run(initial_input="Explain what NEAT does.", answer_only=True)
    for idx, out in enumerate(outputs, start=1):
        print(f"Node {idx} output:\n{out}\n")


if __name__ == "__main__":
    main()
