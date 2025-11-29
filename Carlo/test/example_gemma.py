"""
Example using Gemma 3 270M (HF) as LLM client.
Requires the model checkpoint available locally or downloadable via transformers.
"""
import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Filippo.AgentGenome import AgentGenome
from Filippo.Gene import PromptNode
from Carlo.phenotype import Phenotype
from Carlo.model.llm_gemma import Gemma3Client


def build_sample_genome() -> AgentGenome:
    genome = AgentGenome()

    node1 = PromptNode(name="Checker", instruction="Explain in one sentence what a genetic algorithm is.")
    node2 = PromptNode(
        name="Translator",
        instruction="Traduci in italiano SOLO il testo fornito. Rispondi solo con la traduzione, senza spiegazioni aggiuntive."
    )

    genome.add_node(node1)
    genome.add_node(node2)

    genome.start_node_id = node1.id
    genome.add_connection(node1.id, node2.id)
    return genome


def main():
    genome = build_sample_genome()
    llm = Gemma3Client()
    phenotype = Phenotype(genome, llm)
    outputs = phenotype.run(initial_input="Riassumi il funzionamento di un algoritmo genetico.", answer_only=True)
    for idx, out in enumerate(outputs, start=1):
        print(f"Node {idx} output:\n{out}\n")


if __name__ == "__main__":
    main()
