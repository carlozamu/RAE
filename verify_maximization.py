import sys
import os
import random

# Add Edoardo to path
sys.path.append(os.path.join(os.getcwd(), 'Edoardo'))

from Selection.selection import ElitismSelection, TournamentSelection
from Generation_Manager.generation_manager import CommaPlusStrategy
from Fitness.fitness_function import UnifiedFitnessCalculator

class MockLLM:
    def get_embedding(self, text):
        # Mock embedding, just return random distinct vectors if needed, or zeros
        # For cosine similarity, [1,0] and [1,0] is 1.0
        return [[1.0, 0.0, 0.0]]

def test_fitness_calculation():
    print("\n--- Testing Fitness Calculation (Maximization) ---")
    mock_llm = MockLLM()
    calculator = UnifiedFitnessCalculator(llm=mock_llm, w_accuracy=10.0, w_rationale=5.0, w_token_cost=0.1, w_complexity_cost=0.1)
    
    # Case 1: Perfect Match (Should be high score)
    # Mocking similarity by injecting custom_accuracy since we can't real-run embeddings easily without loading model
    # actually calculator calls self.llm.get_embedding.
    # Let's use custom_accuracy for test
    
    score_perfect = calculator.compute(
        generated_ans="A", target_ans="A",
        custom_accuracy=1.0,
        num_nodes=1, num_edges=0
    )
    print(f"Perfect Score (Acc=1.0): {score_perfect['fitness']}")
    
    # Case 2: Zero Match (Should be low score, but >= 0)
    score_bad = calculator.compute(
        generated_ans="B", target_ans="A",
        custom_accuracy=0.0,
        num_nodes=10, num_edges=20 # High complexity penalty
    )
    print(f"Bad Score (Acc=0.0): {score_bad['fitness']}")
    
    assert score_perfect['fitness'] > score_bad['fitness'], "Perfect score must be higher than bad score"
    assert score_bad['fitness'] >= 0.0, "Fitness must be non-negative"
    print("✅ Fitness Calculation Passed")

def test_selection_maximization():
    print("\n--- Testing Selection (Maximization) ---")
    
    # Population: Higher is better
    population = [
        {'id': 'low', 'fitness': 1.0, 'member': None},
        {'id': 'mid', 'fitness': 5.0, 'member': None},
        {'id': 'high', 'fitness': 10.0, 'member': None}
    ]
    
    # 1. Elitism (Should pick 10.0)
    elitism = ElitismSelection(elite_size=1)
    selected = elitism.select(population, num_parents=1)
    print(f"Elitism Selected: {selected[0]}")
    assert selected[0]['id'] == 'high', f"Elitism failed. Expected 'high', got {selected[0]['id']}"
    
    # 2. Tournament (Should favor higher)
    tournament = TournamentSelection(tournament_size=3)
    # With size 3, it sees all. Max of all is 10.0
    selected_t = tournament.select(population, num_parents=1)
    print(f"Tournament Selected: {selected_t[0]}")
    assert selected_t[0]['id'] == 'high', f"Tournament failed. Expected 'high', got {selected_t[0]['id']}"
    
    print("✅ Selection Strategies Passed")

def test_generation_strategy():
    print("\n--- Testing Generation Strategy (Maximization) ---")
    
    current_pop = [{'id': 'p1', 'fitness': 2.0, 'member': None}, {'id': 'p2', 'fitness': 4.0, 'member': None}]
    offspring_pop = [{'id': 'o1', 'fitness': 8.0, 'member': None}, {'id': 'o2', 'fitness': 1.0, 'member': None}]
    
    # CommaPlus (Selection from both)
    # Should select top N. Top are: 8.0 (o1), 4.0 (p2).
    strategy = CommaPlusStrategy()
    survivors = strategy.select_survivors(current_pop, offspring_pop, population_size=2)
    
    print(f"Survivors: {[s['id'] for s in survivors]}")
    ids = {s['id'] for s in survivors}
    assert 'o1' in ids and 'p2' in ids, f"Generation Strategy failed. Expected o1, p2. Got {ids}"
    
    print("✅ Generation Strategy Passed")

if __name__ == "__main__":
    try:
        test_fitness_calculation()
        test_selection_maximization()
        test_generation_strategy()
        print("\n🎉 ALL TESTS PASSED! Fitness maximization is correctly implemented.")
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        exit(1)
