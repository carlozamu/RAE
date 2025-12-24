"""
Simple example demonstrating how to use the strategy configuration module
for seamless integration with the evolutionary pipeline.
"""
from Edoardo.Selection.strategy_config import StrategyConfig


def example_basic_usage():
    """Basic usage: Create full configuration with one line."""
    print("=" * 80)
    print("Example 1: Basic Usage")
    print("=" * 80)
    
    # Create both managers with recommended strategy
    evolver, gen_manager = StrategyConfig.create_full_config(
        strategy_name="tournament_comma_plus",
        population_size=50,
        num_parents=2
    )
    
    print(f"Created EvolutionManager with: {type(evolver.selection_strategy).__name__}")
    print(f"Created GenerationManager with: {type(gen_manager.survivor_strategy).__name__}")
    print()


def example_all_strategies():
    """Example: Create all three recommended strategies."""
    print("=" * 80)
    print("Example 2: All Three Recommended Strategies")
    print("=" * 80)
    
    strategies = [
        ("tournament_comma_plus", "Tournament + CommaPlus"),
        ("rank_based_comma_plus", "Rank-based + CommaPlus"),
        ("elitism_hall_of_fame", "Elitism + Hall of Fame")
    ]
    
    for strategy_name, description in strategies:
        evolver, gen_manager = StrategyConfig.create_full_config(
            strategy_name=strategy_name,
            population_size=100
        )
        print(f"{description}:")
        print(f"  Selection: {type(evolver.selection_strategy).__name__}")
        print(f"  Survivor: {type(gen_manager.survivor_strategy).__name__}")
        print()


def example_custom_parameters():
    """Example: Customize strategy parameters."""
    print("=" * 80)
    print("Example 3: Custom Parameters")
    print("=" * 80)
    
    # Tournament with larger tournament size
    evolver1, gen_manager1 = StrategyConfig.create_full_config(
        "tournament_comma_plus",
        tournament_size=5,  # Larger tournament
        population_size=100
    )
    print(f"Tournament (size=5): {type(evolver1.selection_strategy).__name__}")
    
    # Rank-based with higher selection pressure
    evolver2, gen_manager2 = StrategyConfig.create_full_config(
        "rank_based_comma_plus",
        selection_pressure=3.0,  # Higher pressure
        population_size=100
    )
    print(f"Rank-based (pressure=3.0): {type(evolver2.selection_strategy).__name__}")
    
    # Elitism with larger hall of fame
    evolver3, gen_manager3 = StrategyConfig.create_full_config(
        "elitism_hall_of_fame",
        elite_size=10,  # More elite individuals
        hall_of_fame_size=20,  # Larger hall of fame
        population_size=100
    )
    print(f"Elitism (elite=10, hof=20): {type(evolver3.selection_strategy).__name__}")
    print()


def example_individual_components():
    """Example: Create components separately."""
    print("=" * 80)
    print("Example 4: Individual Components")
    print("=" * 80)
    
    # Create only EvolutionManager
    evolver = StrategyConfig.create_evolution_manager(
        "rank_based_comma_plus",
        selection_pressure=2.5
    )
    print(f"EvolutionManager: {type(evolver.selection_strategy).__name__}")
    
    # Create only GenerationManager
    gen_manager = StrategyConfig.create_generation_manager(
        "rank_based_comma_plus",
        population_size=100
    )
    print(f"GenerationManager: {type(gen_manager.survivor_strategy).__name__}")
    print()


def example_strategy_objects():
    """Example: Get strategy objects directly."""
    print("=" * 80)
    print("Example 5: Direct Strategy Objects")
    print("=" * 80)
    
    # Get strategy tuple
    selection, survivor = StrategyConfig.get_tournament_comma_plus(
        tournament_size=4
    )
    
    print(f"Selection strategy: {type(selection).__name__}")
    print(f"Survivor strategy: {type(survivor).__name__}")
    print(f"Tournament size: {selection.tournament_size}")
    print()


if __name__ == "__main__":
    example_basic_usage()
    example_all_strategies()
    example_custom_parameters()
    example_individual_components()
    example_strategy_objects()
    
    print("=" * 80)
    print("All examples completed successfully!")
    print("=" * 80)
    print("\nFor pipeline integration, use:")
    print("  evolver, gen_manager = StrategyConfig.create_full_config('tournament_comma_plus')")

