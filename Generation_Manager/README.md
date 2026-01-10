# Generation Manager Module

This module handles generation transitions and survivor selection strategies for evolutionary algorithms.

## Survivor Selection Strategies

### 1. CommaPlusStrategy (μ+λ)
Selects best individuals from both parents and offspring. Preserves best solutions but may slow down exploration.

```python
from Edoardo.Generation_Manager.generation_manager import CommaPlusStrategy, GenerationManager

survivor_strategy = CommaPlusStrategy()
gen_manager = GenerationManager(
    population_size=50,
    survivor_strategy=survivor_strategy
)

next_population = gen_manager.advance_generation(current_population, offspring_population)
```

### 2. CommaStrategy (μ,λ)
Selects survivors only from offspring population. More explorative, forces continuous improvement. Requires λ >= μ.

```python
from Edoardo.Generation_Manager.generation_manager import CommaStrategy

survivor_strategy = CommaStrategy()
gen_manager = GenerationManager(
    population_size=50,
    survivor_strategy=survivor_strategy
)
```

### 3. HallOfFameStrategy
Maintains a separate archive of best-ever individuals. Prevents loss of good solutions found in earlier generations.

```python
from Edoardo.Generation_Manager.generation_manager import HallOfFameStrategy

survivor_strategy = HallOfFameStrategy(hall_of_fame_size=10)
gen_manager = GenerationManager(
    population_size=50,
    survivor_strategy=survivor_strategy
)

# Access hall of fame
hof = survivor_strategy.get_hall_of_fame()
```

## Generation Manager

The `GenerationManager` coordinates generation transitions:

```python
from Edoardo.Generation_Manager.generation_manager import GenerationManager, CommaPlusStrategy

# Initialize
gen_manager = GenerationManager(
    population_size=50,
    survivor_strategy=CommaPlusStrategy(),
    generation=0
)

# Advance generation
next_pop = gen_manager.advance_generation(current_pop, offspring_pop)

# Get statistics
stats = gen_manager.get_statistics()
print(f"Current generation: {stats['current_generation']}")
print(f"Best fitness history: {stats['best_fitness_history']}")
```

## Recommendations

- **CommaPlus (μ+λ)**: Best for most cases, balances exploration and exploitation
- **Comma (μ,λ)**: Use when you want more exploration and can generate many offspring
- **Hall of Fame**: Use when preserving historical best solutions is critical

