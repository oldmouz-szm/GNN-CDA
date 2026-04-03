import networkx as nx
import random

class CircuitSimulator:
    def __init__(self, G):
        """
        Initialize the CircuitSimulator.
        
        Args:
            G (nx.DiGraph): The circuit graph.
        """
        self.G = G
        try:
            self.topo_order = list(nx.topological_sort(G))
        except nx.NetworkXUnfeasible:
            raise ValueError("Circuit has cycles! Topological sort failed.")
            
        # Identify INPUT nodes
        self.input_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'INPUT']

    def _get_logic_value(self, gate_type, input_values):
        """Compute output of a logic gate."""
        if gate_type == 'AND':
            return 1 if all(input_values) else 0
        elif gate_type == 'OR':
            return 1 if any(input_values) else 0
        elif gate_type == 'NAND':
            return 0 if all(input_values) else 1
        elif gate_type == 'NOR':
            return 0 if any(input_values) else 1
        elif gate_type == 'NOT':
            return 1 if not input_values[0] else 0
        elif gate_type == 'BUFF':
            return input_values[0]
        elif gate_type == 'XOR':
            return sum(input_values) % 2
        elif gate_type == 'INPUT':
            return input_values[0]
        else:
            # Default for unknown gates
            return 0

    def simulate(self, input_values, fault=None):
        """
        Simulates the logic circuit.
        
        Args:
            input_values (dict): Map of Input Node Name -> 0/1.
            fault: Can be:
                   - None
                   - tuple (node_name, stuck_at_value) for single fault
                   - list/set of tuples [(n1, v1), (n2, v2)] for multiple faults
            
        Returns:
            dict: State of all nodes {node_name: value}.
        """
        states = {}
        
        # Normalize fault to a dictionary {node: val} for O(1) lookup
        fault_map = {}
        if fault:
            if isinstance(fault, tuple):
                fault_map[fault[0]] = fault[1]
            elif isinstance(fault, (list, set)):
                for f_node, f_val in fault:
                    fault_map[f_node] = f_val
        
        for node in self.topo_order:
            node_type = self.G.nodes[node].get('type', 'UNKNOWN')
            
            # 1. Determine value based on logic
            if node_type == 'INPUT':
                val = input_values.get(node, 0)
            else:
                # Get values of predecessors (inputs to this gate)
                predecessors = list(self.G.predecessors(node))
                input_vals = [states.get(p, 0) for p in predecessors]
                
                if not input_vals and node_type != 'INPUT':
                     val = 0
                else:
                    val = self._get_logic_value(node_type, input_vals)
            
            # 2. Inject Fault (Stuck-at Fault)
            # Check if this node is in the fault map
            if node in fault_map:
                val = fault_map[node]
                
            states[node] = val
            
        return states
