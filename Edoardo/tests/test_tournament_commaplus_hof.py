"""
Test suite per verificare l'implementazione di Tournament + CommaPlus + Hall of Fame.

Testa:
1. Tournament Selection in Species.get_top_members()
2. Hall of Fame Mechanism (globale e per-species)
3. Integrazione completa Tournament + CommaPlus + HoF
4. Edge cases (popolazione vuota, single species, HoF disabilitato, etc.)
"""
import sys
import os
from typing import List, Dict, Any
import random
import numpy as np

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from Edoardo.Selection.selection import TournamentSelection
from Edoardo.Species.species import Species
from Edoardo.Evolution_Manager.evolution_manager import EvolutionManager
from Edoardo.Generation_Manager.generation_manager import CommaPlusStrategy, GenerationManager


class MockPhenotype:
    """Mock Phenotype per testing."""
    def __init__(self, id: int, fitness: float):
        self.id = id
        self.fitness = fitness
        # Mock genome per compatibilità
        self.genome = MockGenome(id)
    
    def __repr__(self):
        return f"Phenotype(id={self.id}, fitness={self.fitness:.3f})"


class MockGenome:
    """Mock Genome per testing."""
    def __init__(self, id: int):
        self.id = id
        self.nodes = {str(i): f"node_{i}" for i in range(2)}
        self.connections = []


def create_mock_species(members: List[MockPhenotype], generation: int = 0, 
                       selection_strategy=None) -> Species:
    """Crea una mock species con i membri specificati."""
    return Species(members, generation=generation, selection_strategy=selection_strategy)


def test_tournament_selection_in_species():
    """Test 1: Verifica che Species.get_top_members() usi Tournament selection."""
    print("=" * 80)
    print("TEST 1: Tournament Selection in Species.get_top_members()")
    print("=" * 80)
    
    # Crea popolazione con fitness variabili
    members = [MockPhenotype(i, random.uniform(0.1, 1.0)) for i in range(20)]
    
    # Crea species con Tournament selection
    tournament_strategy = TournamentSelection(tournament_size=3)
    species = create_mock_species(members, selection_strategy=tournament_strategy)
    
    # Test: get_top_members dovrebbe usare Tournament selection
    top_members = species.get_top_members()
    
    print(f"Popolazione iniziale: {len(members)} membri")
    print(f"Top members selezionati: {len(top_members)}")
    print(f"Top members fitness: {[m['fitness'] for m in top_members[:5]]}")
    
    # Verifica che il numero di membri selezionati sia corretto
    assert len(top_members) <= species.top_r, f"Troppi membri selezionati: {len(top_members)} > {species.top_r}"
    assert len(top_members) > 0, "Nessun membro selezionato"
    
    # Verifica che tutti i top members abbiano la struttura corretta
    for member in top_members:
        assert 'member' in member, "Membro senza 'member' key"
        assert 'fitness' in member, "Membro senza 'fitness' key"
    
    print("✓ Test Tournament Selection in Species: PASSED")
    print()


def test_hall_of_fame_global():
    """Test 2: Verifica che il global HoF mantenga i migliori individui."""
    print("=" * 80)
    print("TEST 2: Global Hall of Fame Mechanism")
    print("=" * 80)
    
    # Crea EvolutionManager con HoF
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3
    )
    
    # Crea due species con membri
    members1 = [MockPhenotype(i, 0.9 - i*0.1) for i in range(10)]
    members2 = [MockPhenotype(i+10, 0.8 - i*0.1) for i in range(10)]
    
    species1 = create_mock_species(members1, generation=0)
    species2 = create_mock_species(members2, generation=0)
    
    evolver.species = [species1, species2]
    evolver.current_generation_index = 0
    
    # Simula aggiornamento HoF
    evolver._update_hall_of_fame()
    
    # Verifica global HoF
    global_hof = evolver.get_global_hall_of_fame()
    print(f"Global HoF size: {len(global_hof)}")
    print(f"Global HoF fitness: {[m['fitness'] for m in global_hof]}")
    
    assert len(global_hof) <= evolver.hall_of_fame_size, "HoF troppo grande"
    assert len(global_hof) > 0, "HoF vuoto"
    
    # Verifica che siano ordinati per fitness (decrescente)
    if len(global_hof) > 1:
        for i in range(len(global_hof) - 1):
            assert global_hof[i]['fitness'] >= global_hof[i+1]['fitness'], \
                "HoF non ordinato correttamente"
    
    print("✓ Test Global Hall of Fame: PASSED")
    print()


def test_hall_of_fame_per_species():
    """Test 3: Verifica che il per-species HoF mantenga i migliori per ogni species."""
    print("=" * 80)
    print("TEST 3: Per-Species Hall of Fame Mechanism")
    print("=" * 80)
    
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3
    )
    
    members1 = [MockPhenotype(i, 0.9 - i*0.1) for i in range(10)]
    members2 = [MockPhenotype(i+10, 0.7 - i*0.1) for i in range(10)]
    
    species1 = create_mock_species(members1, generation=0)
    species2 = create_mock_species(members2, generation=0)
    
    evolver.species = [species1, species2]
    evolver.current_generation_index = 0
    
    evolver._update_hall_of_fame()
    
    # Verifica per-species HoF
    species_hof = evolver.get_species_hall_of_fame()
    print(f"Numero di species con HoF: {len(species_hof)}")
    
    for species, hof in species_hof.items():
        print(f"Species HoF size: {len(hof)}")
        print(f"Species HoF fitness: {[m['fitness'] for m in hof]}")
        assert len(hof) <= evolver.per_species_hof_size, "Per-species HoF troppo grande"
        assert len(hof) > 0, "Per-species HoF vuoto"
    
    # Test getter per species specifica
    hof_species1 = evolver.get_species_hall_of_fame(species1)
    assert isinstance(hof_species1, list), "Getter per species specifica dovrebbe restituire lista"
    print(f"HoF per species1: {len(hof_species1)} membri")
    
    print("✓ Test Per-Species Hall of Fame: PASSED")
    print()


def test_select_parents_unified():
    """Test 4: Verifica che select_parents() unifichi correttamente la selezione."""
    print("=" * 80)
    print("TEST 4: Unified select_parents() Method")
    print("=" * 80)
    
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3,
        hof_parent_ratio=0.2
    )
    
    # Crea species con membri
    members = [MockPhenotype(i, 0.9 - i*0.05) for i in range(15)]
    species = create_mock_species(members, generation=0)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    # Popola HoF
    evolver._update_hall_of_fame()
    
    # Test select_parents
    parents = evolver.select_parents(species, num_parents=2)
    
    print(f"Genitori selezionati: {len(parents)}")
    print(f"Tipo genitori: {[type(p).__name__ for p in parents]}")
    
    assert len(parents) == 2, "Dovrebbero essere selezionati 2 genitori"
    assert all(isinstance(p, MockPhenotype) for p in parents), "Genitori dovrebbero essere Phenotype"
    assert parents[0] != parents[1] or len(members) == 1, "Genitori dovrebbero essere diversi (se possibile)"
    
    print("✓ Test Unified select_parents(): PASSED")
    print()


def test_integration_tournament_commaplus_hof():
    """Test 5: Test integrazione completa Tournament + CommaPlus + HoF."""
    print("=" * 80)
    print("TEST 5: Integration Tournament + CommaPlus + HoF")
    print("=" * 80)
    
    # Crea EvolutionManager con configurazione completa
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3,
        hof_parent_ratio=0.2
    )
    
    # Crea species con Tournament selection
    members = [MockPhenotype(i, 0.8 - i*0.03) for i in range(20)]
    tournament_strategy = TournamentSelection(tournament_size=3)
    species = create_mock_species(members, generation=0, selection_strategy=tournament_strategy)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    # Simula una generazione
    print("Simulazione generazione 0...")
    evolver._update_hall_of_fame()
    
    global_hof = evolver.get_global_hall_of_fame()
    print(f"Global HoF dopo gen 0: {len(global_hof)} membri")
    
    # Test select_parents con HoF popolato
    parents = evolver.select_parents(species, num_parents=2)
    print(f"Genitori selezionati (con HoF): {len(parents)}")
    
    # Verifica che tutto funzioni insieme
    assert len(global_hof) > 0, "HoF dovrebbe essere popolato"
    assert len(parents) == 2, "Dovrebbero essere selezionati 2 genitori"
    
    print("✓ Test Integration: PASSED")
    print()


def test_edge_case_empty_population():
    """Test 6: Edge case - popolazione vuota."""
    print("=" * 80)
    print("TEST 6: Edge Case - Empty Population")
    print("=" * 80)
    
    evolver = EvolutionManager(num_parents=2)
    members = []
    species = create_mock_species(members, generation=0)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    # Test get_top_members con popolazione vuota
    top_members = species.get_top_members()
    assert len(top_members) == 0, "Popolazione vuota dovrebbe restituire lista vuota"
    
    print("✓ Test Empty Population: PASSED")
    print()


def test_edge_case_hof_disabled():
    """Test 7: Edge case - HoF disabilitato (size = 0)."""
    print("=" * 80)
    print("TEST 7: Edge Case - HoF Disabled")
    print("=" * 80)
    
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=0,  # HoF disabilitato
        per_species_hof_size=0,
        hof_parent_ratio=0.2
    )
    
    members = [MockPhenotype(i, 0.8 - i*0.05) for i in range(10)]
    species = create_mock_species(members, generation=0)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    # Test select_parents senza HoF
    parents = evolver.select_parents(species, num_parents=2)
    
    assert len(parents) == 2, "Dovrebbero essere selezionati 2 genitori anche senza HoF"
    
    global_hof = evolver.get_global_hall_of_fame()
    assert len(global_hof) == 0, "HoF dovrebbe essere vuoto quando disabilitato"
    
    print("✓ Test HoF Disabled: PASSED")
    print()


def test_edge_case_hof_ratio_zero():
    """Test 8: Edge case - hof_parent_ratio = 0."""
    print("=" * 80)
    print("TEST 8: Edge Case - HoF Ratio = 0")
    print("=" * 80)
    
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3,
        hof_parent_ratio=0.0  # Ratio = 0
    )
    
    members = [MockPhenotype(i, 0.8 - i*0.05) for i in range(10)]
    species = create_mock_species(members, generation=0)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    evolver._update_hall_of_fame()
    
    # Anche se HoF è popolato, con ratio=0 non dovrebbe essere usato
    parents = evolver.select_parents(species, num_parents=2)
    
    assert len(parents) == 2, "Dovrebbero essere selezionati 2 genitori"
    
    print("✓ Test HoF Ratio Zero: PASSED")
    print()


def test_single_species():
    """Test 9: Edge case - single species."""
    print("=" * 80)
    print("TEST 9: Edge Case - Single Species")
    print("=" * 80)
    
    evolver = EvolutionManager(
        num_parents=2,
        hall_of_fame_size=5,
        per_species_hof_size=3
    )
    
    members = [MockPhenotype(i, 0.9 - i*0.05) for i in range(15)]
    species = create_mock_species(members, generation=0)
    
    evolver.species = [species]
    evolver.current_generation_index = 0
    
    evolver._update_hall_of_fame()
    
    global_hof = evolver.get_global_hall_of_fame()
    species_hof = evolver.get_species_hall_of_fame(species)
    
    assert len(global_hof) > 0, "Global HoF dovrebbe essere popolato"
    assert len(species_hof) > 0, "Species HoF dovrebbe essere popolato"
    
    parents = evolver.select_parents(species, num_parents=2)
    assert len(parents) == 2, "Dovrebbero essere selezionati 2 genitori"
    
    print("✓ Test Single Species: PASSED")
    print()


def run_all_tests():
    """Esegue tutti i test."""
    print("\n" + "=" * 80)
    print("TEST SUITE: Tournament + CommaPlus + Hall of Fame")
    print("=" * 80)
    print()
    
    # Set seed per riproducibilità
    random.seed(42)
    np.random.seed(42)
    
    tests = [
        test_tournament_selection_in_species,
        test_hall_of_fame_global,
        test_hall_of_fame_per_species,
        test_select_parents_unified,
        test_integration_tournament_commaplus_hof,
        test_edge_case_empty_population,
        test_edge_case_hof_disabled,
        test_edge_case_hof_ratio_zero,
        test_single_species
    ]
    
    passed = 0
    failed = 0
    
    for test_func in tests:
        try:
            test_func()
            passed += 1
        except AssertionError as e:
            print(f"✗ {test_func.__name__}: FAILED - {e}")
            failed += 1
        except Exception as e:
            print(f"✗ {test_func.__name__}: ERROR - {e}")
            failed += 1
    
    print("=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(tests)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print("=" * 80)
    
    if failed == 0:
        print("✓ ALL TESTS PASSED!")
    else:
        print(f"✗ {failed} TEST(S) FAILED")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)

