import unittest

from Gene.gene import PromptNode
from Gene.connection import Connection
from Genome.agent_genome import AgentGenome

class TestGenomeTopologicalExecution(unittest.TestCase):

    def setUp(self):
        """Build the default directed graph structure before each test."""
        self.genome = AgentGenome(
            start_node_innovation_number=1, 
            end_node_innovation_number=4
        )
        
        # Add sequential nodes
        self.genome.add_node(PromptNode("Start", "Begin", node_id="1", innovation_number=1))
        self.genome.add_node(PromptNode("Node A", "Process A", node_id="2", innovation_number=2))
        self.genome.add_node(PromptNode("Node B", "Process B", node_id="3", innovation_number=3))
        self.genome.add_node(PromptNode("End", "Finish", node_id="4", innovation_number=4))

        # Build baseline valid DAG (1 -> 2 -> 3 -> 4)
        self.genome.add_connection(1, 2)
        self.genome.add_connection(2, 3)
        self.genome.add_connection(3, 4)

    def test_baseline_execution_order(self):
        """Verify the execution order works perfectly on a valid DAG."""
        plan = self.genome.get_execution_order()
        
        # We expect all 4 nodes to be returned in order
        self.assertEqual(len(plan), 4, "DAG should return full execution plan.")
        self.assertEqual(plan[0][0].innovation_number, 1)
        self.assertEqual(plan[3][0].innovation_number, 4)

    def test_cycle_detection_and_removal(self):
        """Introduce a cycle, prove Kahn's algorithm stalls, and verify resolution."""
        
        # Inject the cycle: Node B (3) points back to Node A (2)
        self.genome.add_connection(3, 2)

        # 1. Prove Kahn's execution stalls gracefully
        stalled_plan = self.genome.get_execution_order()
        self.assertEqual(
            len(stalled_plan), 1, 
            "With a cycle at node 2, its in_degree never hits 0. Execution should halt after Node 1."
        )

        # 2. Detect the cycle
        cycle_edges = self.genome.detect_cycle_edges()
        self.assertTrue(len(cycle_edges) > 0, "Cycle edges should be detected.")

        # 3. Solve the cycle
        disabled_count = self.genome.remove_cycles()
        self.assertEqual(disabled_count, 1, "Exactly one edge should be disabled to resolve the loop.")

        # 4. Verify post-resolution execution order
        resolved_plan = self.genome.get_execution_order()
        self.assertEqual(
            len(resolved_plan), 4, 
            "After cycle resolution, full execution plan should be restored."
        )
        
        # Verify the actual execution order matches the logical flow
        execution_sequence = [step[0].innovation_number for step in resolved_plan]
        self.assertEqual(execution_sequence, [1, 2, 3, 4])

if __name__ == "__main__":
    unittest.main(verbosity=2)
