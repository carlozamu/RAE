"""
Run multiple prompt inputs through a 3-node chain to verify the pipeline.
Uses the stub LLM by default (fast, no network).
"""
import os
import sys

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from Filippo.AgentGenome import AgentGenome
from Filippo.Gene import PromptNode
from Carlo.model.llm_gemma import Gemma3Client
from Carlo.phenotype import Phenotype


def build_genome() -> AgentGenome:
    genome = AgentGenome()

    node1 = PromptNode(
        name="Summarize",
        instruction="Summarize the input in one short sentence (max 20 words)."
    )
    node2 = PromptNode(
        name="Extract",
        instruction=(
            "Convert the summary into exactly two bullet points starting with '- '. "
            "Each bullet must be a concise fact. "
            "Return ONLY the two bullet lines, nothing else."
        )
    )
    node3 = PromptNode(
        name="Polish",
        instruction=(
            "Rewrite the two bullet points in clearer English, keeping the '- ' prefix for each line. "
            "Keep it concise, do not add new facts, and return only the two polished bullets."
        )
    )

    genome.add_node(node1)
    genome.add_node(node2)
    genome.add_node(node3)

    genome.start_node_id = node1.id
    genome.add_connection(node1.id, node2.id)
    genome.add_connection(node2.id, node3.id)
    return genome


def main():
    genome = build_genome()
    llm = Gemma3Client()
    phenotype = Phenotype(genome, llm)

    prompts = [
        "Describe how NEAT handles speciation in neural network evolution.",
        "Explain the main phases of a genetic algorithm applied to prompt optimization.",
        "Summarize what crossover does in a NEAT-like system.",
    ]

    for idx, prompt in enumerate(prompts, start=1):
        outputs = phenotype.run(initial_input=prompt, answer_only=True)
        print(f"\n=== Prompt {idx} ===")
        for j, out in enumerate(outputs, start=1):
            print(f"Node {j} output:\n{out}\n")


if __name__ == "__main__":
    main()
