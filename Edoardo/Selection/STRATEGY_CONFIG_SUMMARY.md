# Strategy Configuration Summary

## Overview

A simple, seamless interface to the three recommended strategy combinations for NEAT-based prompt evolution, based on test results.

## Quick Usage

### One-Line Setup (Recommended)

```python
from Edoardo.Selection.strategy_config import StrategyConfig

# Create both EvolutionManager and GenerationManager
evolver, gen_manager = StrategyConfig.create_full_config(
    strategy_name="tournament_comma_plus",
    population_size=100
)
```

## Three Recommended Strategies

### 1. Tournament + CommaPlus
**Best for**: Good balance of exploration/exploitation

```python
evolver, gen_manager = StrategyConfig.create_full_config("tournament_comma_plus")
```

### 2. Rank-based + CommaPlus  
**Best for**: More robust to fitness scaling

```python
evolver, gen_manager = StrategyConfig.create_full_config("rank_based_comma_plus")
```

### 3. Elitism + Hall of Fame
**Best for**: Preserving top solutions

```python
evolver, gen_manager = StrategyConfig.create_full_config("elitism_hall_of_fame")
```

## Available Methods

### `create_full_config()` - Recommended
Creates both EvolutionManager and GenerationManager with matching strategies.

```python
evolver, gen_manager = StrategyConfig.create_full_config(
    strategy_name="tournament_comma_plus",
    population_size=100,
    num_parents=2,
    tournament_size=3  # Optional parameter
)
```

### `create_evolution_manager()`
Creates only EvolutionManager.

```python
evolver = StrategyConfig.create_evolution_manager(
    "rank_based_comma_plus",
    selection_pressure=2.5
)
```

### `create_generation_manager()`
Creates only GenerationManager.

```python
gen_manager = StrategyConfig.create_generation_manager(
    "elitism_hall_of_fame",
    population_size=100,
    hall_of_fame_size=15
)
```

### Direct Strategy Getters
Get strategy objects as tuples.

```python
selection, survivor = StrategyConfig.get_tournament_comma_plus()
selection, survivor = StrategyConfig.get_rank_based_comma_plus()
selection, survivor = StrategyConfig.get_elitism_hall_of_fame()
```

## Strategy Parameters

### Tournament + CommaPlus
- `tournament_size` (default: 3) - Size of tournament for selection

### Rank-based + CommaPlus
- `selection_pressure` (default: 2.0) - How much better ranks are favored

### Elitism + Hall of Fame
- `elite_size` (default: 5) - Number of top individuals to always select
- `hall_of_fame_size` (default: 10) - Maximum number of best-ever individuals

## Pipeline Integration Example

```python
from Edoardo.Selection.strategy_config import StrategyConfig

# Initialize
evolver, gen_manager = StrategyConfig.create_full_config(
    "tournament_comma_plus",
    population_size=50
)

# In your evolutionary loop
for generation in range(num_generations):
    # Create offspring
    offspring = create_offspring(evolver)
    
    # Advance generation
    next_pop = gen_manager.advance_generation(current_pop, offspring)
    
    # Get statistics
    stats = gen_manager.get_statistics()
```

## Files

- `strategy_config.py` - Main configuration module
- `strategy_config_usage.md` - Detailed usage guide
- `example_strategy_usage.py` - Example code

## Testing Different Strategies

```python
strategies = [
    "tournament_comma_plus",
    "rank_based_comma_plus", 
    "elitism_hall_of_fame"
]

for strategy in strategies:
    evolver, gen_manager = StrategyConfig.create_full_config(strategy)
    # Run your pipeline...
```

