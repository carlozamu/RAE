"""
Strategy Configuration Module

Provides easy access to the three recommended selection and generation management
strategy combinations for NEAT-based prompt evolution.

Based on test results, these three combinations are recommended:
1. Tournament + CommaPlus: Good balance of exploration/exploitation
2. Rank-based + CommaPlus: More robust to fitness scaling
3. Elitism + HallOfFame: Best for preserving top solutions
"""
from typing import Tuple
from Edoardo.Selection.selection import (
    SelectionStrategy,
    TournamentSelection,
    RankBasedSelection,
    ElitismSelection
)
from Edoardo.Generation_Manager.generation_manager import (
    SurvivorSelectionStrategy,
    CommaPlusStrategy,
    HallOfFameStrategy,
    GenerationManager
)
# from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager
import typing
if typing.TYPE_CHECKING:
    from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager


class StrategyConfig:
    """
    Factory class for creating configured selection and generation management strategies.
    """
    
    @staticmethod
    def get_tournament_comma_plus(
        tournament_size: int = 3,
        num_parents: int = 2
    ) -> Tuple[SelectionStrategy, SurvivorSelectionStrategy]:
        """
        Get Tournament selection + CommaPlus survivor strategy.
        
        Best for: Good balance of exploration/exploitation
        
        Args:
            tournament_size: Size of tournament for selection (default: 3)
            num_parents: Number of parents needed (for reference, not used here)
            
        Returns:
            Tuple of (selection_strategy, survivor_strategy)
        """
        selection = TournamentSelection(tournament_size=tournament_size)
        survivor = CommaPlusStrategy()
        return selection, survivor
    
    @staticmethod
    def get_rank_based_comma_plus(
        selection_pressure: float = 2.0,
        num_parents: int = 2
    ) -> Tuple[SelectionStrategy, SurvivorSelectionStrategy]:
        """
        Get Rank-based selection + CommaPlus survivor strategy.
        
        Best for: More robust to fitness scaling issues
        
        Args:
            selection_pressure: Controls how much better ranks are favored (default: 2.0)
            num_parents: Number of parents needed (for reference, not used here)
            
        Returns:
            Tuple of (selection_strategy, survivor_strategy)
        """
        selection = RankBasedSelection(selection_pressure=selection_pressure)
        survivor = CommaPlusStrategy()
        return selection, survivor
    
    @staticmethod
    def get_elitism_hall_of_fame(
        elite_size: int = 5,
        hall_of_fame_size: int = 10,
        num_parents: int = 2
    ) -> Tuple[SelectionStrategy, SurvivorSelectionStrategy]:
        """
        Get Elitism selection + Hall of Fame survivor strategy.
        
        Best for: Preserving top solutions, preventing loss of best-ever individuals
        
        Args:
            elite_size: Number of top individuals to always select (default: 5)
            hall_of_fame_size: Maximum number of best-ever individuals to maintain (default: 10)
            num_parents: Number of parents needed (for reference, not used here)
            
        Returns:
            Tuple of (selection_strategy, survivor_strategy)
        """
        selection = ElitismSelection(elite_size=elite_size)
        survivor = HallOfFameStrategy(hall_of_fame_size=hall_of_fame_size)
        return selection, survivor
    
    @staticmethod
    def create_evolution_manager(
        strategy_name: str = "tournament_comma_plus",
        num_parents: int = 2,
        **kwargs
    ) -> 'EvolutionManager':
        """
        Create a configured EvolutionManager with one of the recommended strategies.
        
        Args:
            strategy_name: One of "tournament_comma_plus", "rank_based_comma_plus", 
                         or "elitism_hall_of_fame"
            num_parents: Number of parents needed for reproduction
            **kwargs: Additional parameters for strategy configuration:
                - tournament_size (for tournament_comma_plus, default: 3)
                - selection_pressure (for rank_based_comma_plus, default: 2.0)
                - elite_size (for elitism_hall_of_fame, default: 5)
                - hall_of_fame_size (for elitism_hall_of_fame, default: 10)
        
        Returns:
            Configured EvolutionManager instance
        """
        strategy_map = {
            "tournament_comma_plus": StrategyConfig.get_tournament_comma_plus,
            "rank_based_comma_plus": StrategyConfig.get_rank_based_comma_plus,
            "elitism_hall_of_fame": StrategyConfig.get_elitism_hall_of_fame
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Choose from: {list(strategy_map.keys())}"
            )
        
        from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager
        # Extract HoF parameters from kwargs if present, or use defaults
        per_species_hof_size = kwargs.get('per_species_hof_size', 10)
        hof_parent_ratio = kwargs.get('hof_parent_ratio', 0.2)
        
        selection_strategy, survivor_strategy = strategy_map[strategy_name](num_parents=num_parents, **kwargs)
        return EvolutionManager(
            selection_strategy=selection_strategy,
            survivor_strategy=survivor_strategy,
            num_parents=num_parents,
            per_species_hof_size=per_species_hof_size,
            hof_parent_ratio=hof_parent_ratio
        )
    
    @staticmethod
    def create_generation_manager(
        strategy_name: str = "tournament_comma_plus",
        population_size: int = 50,
        generation: int = 0,
        **kwargs
    ) -> GenerationManager:
        """
        Create a configured GenerationManager with one of the recommended strategies.
        
        Args:
            strategy_name: One of "tournament_comma_plus", "rank_based_comma_plus", 
                         or "elitism_hall_of_fame"
            population_size: Target size for each generation
            generation: Starting generation number
            **kwargs: Additional parameters for strategy configuration:
                - tournament_size (for tournament_comma_plus, default: 3)
                - selection_pressure (for rank_based_comma_plus, default: 2.0)
                - elite_size (for elitism_hall_of_fame, default: 5)
                - hall_of_fame_size (for elitism_hall_of_fame, default: 10)
        
        Returns:
            Configured GenerationManager instance
            
        Example:
            >>> gen_manager = StrategyConfig.create_generation_manager(
            ...     "tournament_comma_plus",
            ...     population_size=100
            ... )
        """
        strategy_map = {
            "tournament_comma_plus": StrategyConfig.get_tournament_comma_plus,
            "rank_based_comma_plus": StrategyConfig.get_rank_based_comma_plus,
            "elitism_hall_of_fame": StrategyConfig.get_elitism_hall_of_fame
        }
        
        if strategy_name not in strategy_map:
            raise ValueError(
                f"Unknown strategy: {strategy_name}. "
                f"Choose from: {list(strategy_map.keys())}"
            )
        
        _, survivor_strategy = strategy_map[strategy_name](**kwargs)
        return GenerationManager(
            population_size=population_size,
            survivor_strategy=survivor_strategy,
            generation=generation
        )
    
    @staticmethod
    def create_full_config(
        strategy_name: str = "tournament_comma_plus",
        num_parents: int = 2,
        population_size: int = 50,
        generation: int = 0,
        **kwargs
    ) -> typing.Tuple['EvolutionManager', GenerationManager]:
        """
        Create both EvolutionManager and GenerationManager with matching strategies.
        
        This is the recommended way to set up the full evolutionary pipeline.
        
        Args:
            strategy_name: One of "tournament_comma_plus", "rank_based_comma_plus", 
                         or "elitism_hall_of_fame"
            num_parents: Number of parents needed for reproduction
            population_size: Target size for each generation
            generation: Starting generation number
            **kwargs: Additional parameters for strategy configuration
        
        Returns:
            Tuple of (EvolutionManager, GenerationManager)
            
        Example:
            >>> evolver, gen_manager = StrategyConfig.create_full_config(
            ...     "tournament_comma_plus",
            ...     population_size=100
            ... )
        """
        evolver = StrategyConfig.create_evolution_manager(
            strategy_name=strategy_name,
            num_parents=num_parents,
            **kwargs
        )
        gen_manager = StrategyConfig.create_generation_manager(
            strategy_name=strategy_name,
            population_size=population_size,
            generation=generation,
            **kwargs
        )
        return evolver, gen_manager


# Convenience functions for direct access
def get_tournament_comma_plus(**kwargs):
    """Convenience function for Tournament + CommaPlus strategy."""
    return StrategyConfig.get_tournament_comma_plus(**kwargs)


def get_rank_based_comma_plus(**kwargs):
    """Convenience function for Rank-based + CommaPlus strategy."""
    return StrategyConfig.get_rank_based_comma_plus(**kwargs)


def get_elitism_hall_of_fame(**kwargs):
    """Convenience function for Elitism + Hall of Fame strategy."""
    return StrategyConfig.get_elitism_hall_of_fame(**kwargs)

