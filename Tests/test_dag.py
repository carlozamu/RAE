import os
import sys
import unittest
import asyncio

# Get the absolute path of the directory one level up (RAE)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Append it to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from Gene.gene import PromptNode
from Genome.agent_genome import AgentGenome
from Mutations.mutator import Mutator
from Utils.LLM import LLM

class TestAgentGenomeDAG(unittest.TestCase):
    
    def setUp(self):
        # Explicitly initialize nodes matching production signature: (name, instruction, ..., innovation_number)
        n1 = PromptNode(name="Start", instruction="Init query", innovation_number=1)
        n2 = PromptNode(name="Mid A", instruction="Task split A", innovation_number=2)
        n3 = PromptNode(name="Mid B", instruction="Task split B", innovation_number=3)
        n4 = PromptNode(name="End", instruction="Compile output", innovation_number=4)
        
        self.genome = AgentGenome(start_node_innovation_number=1, end_node_innovation_number=4)
        self.genome.add_node(n1)
        self.genome.add_node(n2)
        self.genome.add_node(n3)
        self.genome.add_node(n4)
        
        self.genome.add_connection(1, 2)
        self.genome.add_connection(1, 3)
        self.genome.add_connection(2, 4)
        self.genome.add_connection(3, 4)

        llm = LLM()
        self.mutator = Mutator(llm)

    def test_verify_all_paths_valid_dag(self):
        """Test that the baseline DAG topology evaluates as strictly valid."""
        self.assertTrue(self.genome.verify_all_paths_lead_to_end())

    def test_detect_dead_end(self):
        """Test that adding a node without an active path out to END invalidates the graph."""
        n5 = PromptNode(name="Dead Node", instruction="Stray processing", innovation_number=5)
        self.genome.add_node(n5)
        self.genome.add_connection(2, 5) 
        # Node 5 has an incoming edge from 2, but goes nowhere. This must fail graph validation.
        self.assertFalse(self.genome.verify_all_paths_lead_to_end())

    def test_detect_orphan(self):
        """Test that adding an unrouted node without a path from START invalidates the graph."""
        n6 = PromptNode(name="Orphan Node", instruction="Dangling task", innovation_number=6)
        self.genome.add_node(n6)
        self.genome.add_connection(6, 4) 
        # Node 6 reaches End, but cannot be reached by Start. This must fail graph validation.
        self.assertFalse(self.genome.verify_all_paths_lead_to_end())

    def test_execution_order_topological_sort(self):
        """Ensure Kahn's algorithm outputs correct dependency ordering."""
        order = self.genome.get_execution_order()
        node_ids = [n.innovation_number for n, _ in order]
        
        self.assertEqual(node_ids[0], 1)  # Start Node first
        self.assertEqual(node_ids[-1], 4) # End Node last
        self.assertIn(2, node_ids[1:3])   # Intermediate evaluation steps
        self.assertIn(3, node_ids[1:3])

    def test_safe_node_removal_does_not_leave_dangling_edges(self):
        """Verify structural node deletion thoroughly scrubs the connection dictionary keys."""
        self.genome.remove_node_safely(2)
        
        self.assertNotIn(2, self.genome.nodes)
        # Check string IDs matching connection pattern 'in.out'
        self.assertNotIn("1.2", self.genome.connections)
        self.assertNotIn("2.4", self.genome.connections)

    def test_transactional_remove_node_heals_graph(self):
        """Verify removing an intermediate node cleanly bridges remaining paths."""
        self.mutator._handle_remove_node(self.genome)
        self.assertTrue(self.genome.verify_all_paths_lead_to_end())
        self.assertEqual(len(self.genome.nodes), 3)

    def test_transactional_remove_connection_reverts_critical_breaks(self):
        """Verify mutations targeting single structural bridge connections roll back completely."""
        linear_genome = AgentGenome(start_node_innovation_number=1, end_node_innovation_number=3)
        ln1 = PromptNode(name="S", instruction="I", innovation_number=1)
        ln2 = PromptNode(name="M", instruction="P", innovation_number=2)
        ln3 = PromptNode(name="E", instruction="C", innovation_number=3)
        
        linear_genome.add_node(ln1)
        linear_genome.add_node(ln2)
        linear_genome.add_node(ln3)
        linear_genome.add_connection(1, 2)
        linear_genome.add_connection(2, 3)
        
        # Deleting either link cuts off the pipeline completely
        self.mutator._handle_remove_connection(linear_genome)
        
        # Transaction must detect the break, trigger a rollback, and retain stability
        self.assertTrue(linear_genome.verify_all_paths_lead_to_end())
        active_edges = [c for c in linear_genome.connections.values() if c.enabled]
        self.assertEqual(len(active_edges), 2)

if __name__ == '__main__':
    unittest.main()
