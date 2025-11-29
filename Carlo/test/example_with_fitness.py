"""
Example: run a 2-node chain with Gemma and score the final output with the Devvone fitness function.
"""
import os
import sys
import time

# Ensure repo root is on sys.path
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import evaluate
import Devvone.fitness_function as fitness_mod
from Devvone.fitness_function import CostCalculator
from Filippo.AgentGenome import AgentGenome
from Filippo.Gene import PromptNode
from Carlo.model.llm_gemma import Gemma3Client
from Carlo.phenotype import Phenotype


def build_sample_genome() -> AgentGenome:
    genome = AgentGenome()
    node1 = PromptNode(name="Summarize", instruction="Summarize the input in one concise sentence.")
    node2 = PromptNode(name="Answer", instruction="Give a direct answer in one short sentence.")
    genome.add_node(node1)
    genome.add_node(node2)
    genome.start_node_id = node1.id
    genome.add_connection(node1.id, node2.id)
    return genome


def main():
    # Init rouge for the fitness module (it expects a global rouge_metric)
    fitness_mod.rouge_metric = evaluate.load("rouge")
    calculator = CostCalculator(
        w_accuracy=10.0,
        w_token_cost=0.05,
        w_time_cost=0.2,
        w_divergence=1.0,
    )

    # Simple prompt/target pairs for scoring
    samples = [
        {
            "prompt": "What does NEAT do in neural network evolution?",
            "target": "It evolves neural network topology and weights using neuroevolution.",
        },
        {
            "prompt": "What is the main role of crossover in a genetic algorithm?",
            "target": "It combines genetic material from parents to create offspring.",
        },
    ]

    genome = build_sample_genome()
    llm = Gemma3Client()
    phenotype = Phenotype(genome, llm)

    for idx, sample in enumerate(samples, start=1):
        start_time = time.time()
        outputs = phenotype.run(initial_input=sample["prompt"], answer_only=True)
        elapsed = time.time() - start_time
        generated = outputs[-1] if outputs else ""
        score = calculator.compute(generated_text=generated, target_text=sample["target"], generation_time=elapsed)
        print(f"\n=== Sample {idx} ===")
        print(f"Prompt: {sample['prompt']}")
        print(f"Generated: {generated}")
        print(f"Target: {sample['target']}")
        print(f"Fitness (lower is better): {score['loss']}  Details: {score['details']}")


if __name__ == "__main__":
    main()
