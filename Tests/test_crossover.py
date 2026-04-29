import unittest
from unittest.mock import patch
import numpy as np

from Genome.agent_genome import AgentGenome
from Gene.gene import PromptNode
from Gene.connection import Connection
from Crossover.crossover import Crossover 

class TestCrossoverEdgeCases(unittest.TestCase):

    def setUp(self):
        """Helper to set up basic identical parents for baseline manipulation."""
        self.p1 = AgentGenome(start_node_innovation_number=1, end_node_innovation_number=4)
        self.p2 = AgentGenome(start_node_innovation_number=1, end_node_innovation_number=4)
        
        # Shared Start and End nodes
        self.p1.add_node(PromptNode("Start", "Begin P1", node_id="1", innovation_number=1))
        self.p2.add_node(PromptNode("Start", "Begin P2", node_id="1", innovation_number=1))
        
        self.p1.add_node(PromptNode("End", "Finish P1", node_id="4", innovation_number=4))
        self.p2.add_node(PromptNode("End", "Finish P2", node_id="4", innovation_number=4))

    def test_unequal_fitness_drops_worse_disjoint_genes(self):
        """
        Rule: If P1 > P2, offspring must NOT inherit disjoint/excess genes from P2.
        """
        self.p1.fitness = 10.0
        self.p2.fitness = 5.0
        
        # P1 has Node 2 and conn 1->2->4
        self.p1.add_node(PromptNode("A", "P1 Node", node_id="2", innovation_number=2))
        self.p1.add_connection(1, 2)
        self.p1.add_connection(2, 4)
        
        # P2 has Node 3 and conn 1->3->4 (This is disjoint and P2 is worse)
        self.p2.add_node(PromptNode("B", "P2 Node", node_id="3", innovation_number=3))
        self.p2.add_connection(1, 3)
        self.p2.add_connection(3, 4)

        child = Crossover.create_offspring(self.p1, self.p2)

        self.assertIn(2, child.nodes, "Child should inherit better parent's disjoint node.")
        self.assertNotIn(3, child.nodes, "Child MUST drop worse parent's disjoint node.")
        
        # Verify connections based on nodes (generate hashes to check)
        conn_1_2_hash = Connection(1, 2).innovation_number
        conn_1_3_hash = Connection(1, 3).innovation_number
        
        self.assertIn(conn_1_2_hash, child.connections)
        self.assertNotIn(conn_1_3_hash, child.connections)

    def test_equal_fitness_unions_disjoint_genes(self):
        """
        Rule: If P1 == P2, offspring inherits disjoint genes from BOTH parents.
        """
        self.p1.fitness = 5.0
        self.p2.fitness = 5.0
        
        # P1 has Node 2 (1->2->4)
        self.p1.add_node(PromptNode("A", "P1 Node", node_id="2", innovation_number=2))
        self.p1.add_connection(1, 2)
        self.p1.add_connection(2, 4)
        
        # P2 has Node 3 (1->3->4)
        self.p2.add_node(PromptNode("B", "P2 Node", node_id="3", innovation_number=3))
        self.p2.add_connection(1, 3)
        self.p2.add_connection(3, 4)

        child = Crossover.create_offspring(self.p1, self.p2)

        # Child should have both Node 2 and Node 3
        self.assertIn(2, child.nodes)
        self.assertIn(3, child.nodes)

    @patch('numpy.random.rand')
    def test_semantic_node_inheritance(self, mock_rand):
        self.p1.fitness = 5.0
        self.p2.fitness = 5.0
        
        # Ensure nodes 1 and 4 are connected so they aren't pruned
        self.p1.add_connection(1, 4)
        self.p2.add_connection(1, 4)
        
        # TEST 1: Force "low" rolls (always < 0.5) to pick P1 traits
        mock_rand.side_effect = lambda: 0.1 
        child_p1_trait = Crossover.create_offspring(self.p1, self.p2)
        self.assertEqual(child_p1_trait.nodes[1].instruction, "Begin P1")

        # TEST 2: Force "high" rolls (always > 0.5) to pick P2 traits
        mock_rand.side_effect = lambda: 0.9 
        child_p2_trait = Crossover.create_offspring(self.p1, self.p2)
        self.assertEqual(child_p2_trait.nodes[1].instruction, "Begin P2")

    @patch('numpy.random.rand')
    def test_75_percent_disable_rule(self, mock_rand):
        self.p1.fitness = 5.0
        self.p2.fitness = 5.0
        
        # Create a "Backbone" connection that is ALWAYS enabled
        # This prevents nodes 1 and 4 from being pruned
        self.p1.add_node(PromptNode("Aux", "Aux", node_id="99", innovation_number=99))
        self.p2.add_node(PromptNode("Aux", "Aux", node_id="99", innovation_number=99))
        self.p1.add_connection(1, 99); self.p1.add_connection(99, 4)
        self.p2.add_connection(1, 99); self.p2.add_connection(99, 4)

        # Now add the connection we actually want to test
        self.p1.add_connection(1, 4)
        self.p2.add_connection(1, 4)
        conn_hash = Connection(1, 4).innovation_number
        
        self.p1.connections[conn_hash].enabled = False
        self.p2.connections[conn_hash].enabled = True

        # Side effect needs to account for the extra nodes and connections
        # We just need to make sure the roll for our target conn_hash is controlled
        # Let's simplify: mock_rand.return_value = 0.5 for a specific call 
        # or just provide a long list of 0.1s and one 0.5
        mock_rand.side_effect = [0.1] * 20 
        
        # Manually find which roll index corresponds to the disable check or 
        # just use a list that covers all possible rolls.
        child_disabled = Crossover.create_offspring(self.p1, self.p2)
        self.assertIn(conn_hash, child_disabled.connections)
        self.assertFalse(child_disabled.connections[conn_hash].enabled)

    def test_topological_pruning(self):
        """
        Rule: Pruning must remove nodes/edges that result in dead ends or unreachable branches 
        created during equal-fitness crossover union.
        """
        self.p1.fitness = 5.0
        self.p2.fitness = 5.0
        
        # P1 establishes a valid path: 1 -> 2 -> 4(End)
        self.p1.add_node(PromptNode("A", "Valid Node", node_id="2", innovation_number=2))
        self.p1.add_connection(1, 2)
        self.p1.add_connection(2, 4)
        
        # P2 has a dead-end branch: 1 -> 3 (Does NOT reach 4)
        self.p2.add_node(PromptNode("B", "Dead End Node", node_id="3", innovation_number=3))
        self.p2.add_connection(1, 3)

        # Because fitness is equal, crossover will attempt to union everything.
        # This creates a child with a valid path AND a dead end at Node 3.
        child = Crossover.create_offspring(self.p1, self.p2)

        # The _prune_invalid_topology method MUST detect that Node 3 cannot reach Node 4 and delete it.
        self.assertIn(2, child.nodes, "Valid path node should be preserved.")
        self.assertNotIn(3, child.nodes, "Dead-end node MUST be pruned.")
        
        conn_1_2_hash = Connection(1, 2).innovation_number
        conn_1_3_hash = Connection(1, 3).innovation_number
        
        self.assertIn(conn_1_2_hash, child.connections, "Valid connections should be preserved.")
        self.assertNotIn(conn_1_3_hash, child.connections, "Connections leading to pruned nodes MUST be deleted.")

if __name__ == '__main__':
    unittest.main(verbosity=2)