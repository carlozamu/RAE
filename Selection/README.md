# Selection Module

This module provides various selection strategies for evolutionary algorithms, specifically designed for NEAT-based prompt evolution.

## Selection Strategies

### 1. ElitismSelection
Always selects the top N individuals by fitness. Preserves the best individuals across generations.

```python
from Edoardo.Selection.selection import ElitismSelection

selector = ElitismSelection(elite_size=5)
parents = selector.select(population, num_parents=10)
```

### 2. RankBasedSelection
Assigns selection probability based on rank rather than raw fitness. More robust to fitness scaling issues.

```python
from Edoardo.Selection.selection import RankBasedSelection

selector = RankBasedSelection(selection_pressure=2.0)
parents = selector.select(population, num_parents=10)
```

### 3. TournamentSelection
Randomly selects k individuals and picks the best from each tournament. Simple, effective, and maintains diversity.

```python
from Edoardo.Selection.selection import TournamentSelection

selector = TournamentSelection(tournament_size=3)
parents = selector.select(population, num_parents=10)
```

### 4. FitnessProportionateSelection
Selection probability proportional to fitness (Roulette Wheel). Works well when fitness values are positive and well-scaled.

```python
from Edoardo.Selection.selection import FitnessProportionateSelection

selector = FitnessProportionateSelection(use_adjusted_fitness=False)
parents = selector.select(population, num_parents=10)
```

## Population Format

All selection strategies expect populations in the following format:
```python
population = [
    {'member': phenotype_object, 'fitness': 0.85},
    {'member': phenotype_object, 'fitness': 0.72},
    # ...
]
```

## Recommendations for NEAT

For NEAT-based prompt evolution:
- **Tournament Selection**: Good balance, widely used, maintains diversity
- **Rank-based Selection**: More robust to fitness scaling, good for complex fitness landscapes
- **Elitism**: Best when you want to guarantee preservation of top solutions

