import os
import sys

import pytest

# Get the absolute path of the directory one level up (RAE)
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)

# Append it to sys.path if it's not already there
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
    
from Gene.gene import PromptNode
from Gene.connection import Connection
from Genome.agent_genome import AgentGenome

# ==========================================
# Test Fixtures (Graph Setups)
# ==========================================

@pytest.fixture
def base_linear_genome():
    """
    Creates a minimal, perfectly stable linear graph:
    Start(1) -> End(2)
    """
    node_start = PromptNode("Start", "System Prompt", innovation_number=1)
    node_end = PromptNode("End", "Output Result", innovation_number=2)
    
    genome = AgentGenome(
        nodes_dict={1: node_start, 2: node_end},
        connections_dict={},
        start_node_innovation_number=1,
        end_node_innovation_number=2
    )
    
    # Wire the initial linear connection manually for the baseline
    conn = Connection(1, 2)
    genome.connections[conn.innovation_number] = conn
    
    return genome

# ==========================================
# Test Suite
# ==========================================

def test_verify_baseline_integrity(base_linear_genome):
    """Ensure the baseline graph passes topological verification."""
    assert base_linear_genome.verify_all_paths_lead_to_end() is True

def test_add_node_safely_splits_edge(base_linear_genome):
    """
    Edge Case: Adding a node must split a connection to prevent orphans.
    Graph Transition: (1 -> 2) becomes (1 -> 3 -> 2)
    """
    node_3 = PromptNode("Mid", "Think", innovation_number=3)
    target_conn_id = "1.2"
    
    success = base_linear_genome.add_node_safely(node_3, target_conn_id)
    
    assert success is True
    assert 3 in base_linear_genome.nodes
    assert "1.2" not in base_linear_genome.connections
    assert "1.3" in base_linear_genome.connections
    assert "3.2" in base_linear_genome.connections
    assert base_linear_genome.verify_all_paths_lead_to_end() is True

def test_add_node_safely_invalid_target(base_linear_genome):
    """Edge Case: Adding a node to a non-existent connection should fail."""
    node_3 = PromptNode("Mid", "Think", innovation_number=3)
    success = base_linear_genome.add_node_safely(node_3, "9.9") # Non-existent
    assert success is False
    assert 3 not in base_linear_genome.nodes

def test_boundary_strictness_enforcement(base_linear_genome):
    """
    Edge Case: The start node cannot have incoming connections. 
    The end node cannot have outgoing connections.
    """
    # 1. Attempt to route End -> Start
    conn1 = base_linear_genome.add_connection_safely(2, 1)
    assert conn1 is None
    
    # 2. Attempt to route End -> New Node
    node_3 = PromptNode("Mid", "Think", innovation_number=3)
    base_linear_genome.nodes[3] = node_3
    conn2 = base_linear_genome.add_connection_safely(2, 3)
    assert conn2 is None

def test_acyclicity_enforcement(base_linear_genome):
    """
    Edge Case: Prevent cycles from forming.
    Graph: 1 -> 3 -> 4 -> 2. Attempt to route 4 -> 3.
    """
    node_3 = PromptNode("Mid1", "", innovation_number=3)
    node_4 = PromptNode("Mid2", "", innovation_number=4)
    
    # Build: 1 -> 3 -> 4 -> 2
    base_linear_genome.add_node_safely(node_3, "1.2")
    base_linear_genome.add_node_safely(node_4, "3.2")
    
    # Attempt to route backward: 4 -> 3
    bad_conn = base_linear_genome.add_connection_safely(4, 3)
    assert bad_conn is None, "Failed to catch deep cycle creation"
    
    # Attempt self-loop: 3 -> 3
    bad_conn_2 = base_linear_genome.add_connection_safely(3, 3)
    assert bad_conn_2 is None, "Failed to catch self-loop cycle"

def test_remove_connection_transactional_revert(base_linear_genome):
    """
    Edge Case: Deleting a load-bearing connection should trigger a revert.
    Graph: 1 -> 2. Deleting 1.2 creates a severed graph.
    """
    success = base_linear_genome.remove_connection_safely("1.2")
    
    assert success is False, "Allowed deletion of critical path"
    assert "1.2" in base_linear_genome.connections, "Revert failed: Connection was lost"
    assert base_linear_genome.verify_all_paths_lead_to_end() is True

def test_remove_connection_safe_redundancy(base_linear_genome):
    """
    Edge Case: Deleting a redundant parallel path should succeed.
    Graph: 1 -> 3 -> 2, and 1 -> 2. Deleting 1.2 is safe.
    """
    node_3 = PromptNode("Mid", "", innovation_number=3)
    base_linear_genome.nodes[3] = node_3
    
    # Manually wire 1 -> 3 and 3 -> 2 alongside the existing 1 -> 2
    base_linear_genome.connections["1.3"] = Connection(1, 3)
    base_linear_genome.connections["3.2"] = Connection(3, 2)
    
    assert base_linear_genome.verify_all_paths_lead_to_end() is True
    
    # Deleting the direct 1.2 parallel path is safe because 1->3->2 still exists
    success = base_linear_genome.remove_connection_safely("1.2")
    assert success is True
    assert "1.2" not in base_linear_genome.connections

def test_remove_node_safely_boundary_protection(base_linear_genome):
    """Edge Case: Start and End nodes must be immutable."""
    transaction_start = base_linear_genome.remove_node_safely(1)
    transaction_end = base_linear_genome.remove_node_safely(2)
    
    assert transaction_start["removed_node"] is None
    assert transaction_end["removed_node"] is None
    assert 1 in base_linear_genome.nodes
    assert 2 in base_linear_genome.nodes

def test_remove_node_safely_cartesian_bypass(base_linear_genome):
    """
    Edge Case: Deleting a central hub node must correctly wire all sources to all targets.
    Graph: (1->3), (4->3), (3->2), (3->5)
    Delete 3. Expected bypasses: 1->2, 1->5, 4->2, 4->5
    """
    # Note: We manually build this graph bypassing standard safe methods 
    # purely to test the remove_node bypass logic mathematically.
    genome = AgentGenome(
        nodes_dict={
            1: PromptNode("S", "", innovation_number=1),
            2: PromptNode("E", "", innovation_number=2),
            3: PromptNode("Hub", "", innovation_number=3),
            4: PromptNode("In2", "", innovation_number=4),
            5: PromptNode("Out2", "", innovation_number=5)
        },
        connections_dict={
            "1.3": Connection(1, 3),
            "4.3": Connection(4, 3),
            "3.2": Connection(3, 2),
            "3.5": Connection(3, 5)
        },
        start_node_innovation_number=1,
        end_node_innovation_number=2
    )
    
    transaction = genome.remove_node_safely(3)
    
    # Verify node and old connections are gone
    assert transaction["removed_node"] is not None
    assert 3 not in genome.nodes
    assert "1.3" not in genome.connections
    assert "4.3" not in genome.connections
    
    # Verify exact Cartesian bypass
    assert "1.2" in genome.connections
    assert "1.5" in genome.connections
    assert "4.2" in genome.connections
    assert "4.5" in genome.connections
    
    # Verify transaction log caught everything for potential rollback
    assert len(transaction["removed_connections"]) == 4
    assert len(transaction["added_connections"]) == 4

def test_topological_sort_order(base_linear_genome):
    """Verify execution order respects dependencies."""
    node_3 = PromptNode("Mid1", "", innovation_number=3)
    node_4 = PromptNode("Mid2", "", innovation_number=4)
    
    # Build: 1 -> 3 -> 4 -> 2
    base_linear_genome.add_node_safely(node_3, "1.2")
    base_linear_genome.add_node_safely(node_4, "3.2")
    
    execution_plan = base_linear_genome.get_execution_order()
    
    # Extract just the node IDs for easy checking
    order = [node.innovation_number for node, _ in execution_plan]
    assert order == [1, 3, 4, 2]