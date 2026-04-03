import torch
import networkx as nx
from torch_geometric.data import Data
import re
import os

def parse_bench_to_networkx(file_path):
    """
    Parses a .bench file into a NetworkX DiGraph.
    
    Args:
        file_path (str): Path to the .bench file.
        
    Returns:
        nx.DiGraph: A directed graph representing the circuit.
                    Nodes have a 'type' attribute.
    """
    G = nx.DiGraph()
    
    # Regex patterns
    # Matches: INPUT(G1)
    input_pattern = re.compile(r'INPUT\s*\((.+?)\)')
    # Matches: G1 = AND(G2, G3)
    gate_pattern = re.compile(r'(\S+)\s*=\s*(\w+)\s*\((.+?)\)')
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                
                # Check for INPUT
                input_match = input_pattern.match(line)
                if input_match:
                    node_name = input_match.group(1)
                    G.add_node(node_name, type='INPUT')
                    continue
                
                # Check for Gate definitions
                # Format: Output = TYPE(Input1, Input2, ...)
                gate_match = gate_pattern.match(line)
                if gate_match:
                    output_node = gate_match.group(1)
                    gate_type = gate_match.group(2).upper()
                    inputs_str = gate_match.group(3)
                    
                    # Handle inputs separated by commas
                    inputs = [x.strip() for x in inputs_str.split(',')]
                    
                    G.add_node(output_node, type=gate_type)
                    
                    for input_node in inputs:
                        # Add edge from input to output (signal flow)
                        G.add_edge(input_node, output_node)
                    continue
                    
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        return None

    return G

def convert_to_pyg_data(nx_graph, simulation_data=None):
    """
    Converts NetworkX graph to PyG Data object.
    
    Args:
        nx_graph: NetworkX DiGraph
        simulation_data: Dictionary mapping node names to state info.
                         To support the 3 dynamic features [expected, observed, is_diff],
                         the values in this dictionary should ideally be a dict or tuple:
                         {'expected': v1, 'observed': v2} or (v1, v2).
                         If a single value is provided, it is treated as 'observed' and 'expected' is -1.
    
    Returns:
        torch_geometric.data.Data: PyG Data object with x, edge_index, and node_names.
    """
    if nx_graph is None:
        return None

    # 1. Node Mapping
    # Sort nodes to ensure deterministic ordering
    node_names = sorted(list(nx_graph.nodes()))
    node_map = {name: i for i, name in enumerate(node_names)}
    num_nodes = len(node_names)
    
    # 2. Edge Index
    # Create forward edges
    edge_indices = []
    for u, v in nx_graph.edges():
        if u in node_map and v in node_map:
            edge_indices.append([node_map[u], node_map[v]])
            
    # Convert to tensor
    if len(edge_indices) > 0:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    # Add reverse edges (bidirectional)
    # This allows GNNs to pass messages in both directions (e.g. for diagnosis)
    if edge_index.numel() > 0:
        row, col = edge_index
        # Concatenate original edges with reversed edges (col, row)
        edge_index = torch.cat([edge_index, torch.stack([col, row], dim=0)], dim=1)
        # Remove duplicates to ensure clean graph
        edge_index = torch.unique(edge_index, dim=1)

    # 3. Node Features
    # Part 1: One-hot encoding of gate types
    gate_types = ['INPUT', 'AND', 'OR', 'NAND', 'NOR', 'NOT', 'BUFF', 'XOR']
    type_to_idx = {t: i for i, t in enumerate(gate_types)}
    
    x_features = []
    
    for name in node_names:
        node_attrs = nx_graph.nodes[name]
        g_type = node_attrs.get('type', 'UNKNOWN')
        
        # One-hot vector
        one_hot = [0] * len(gate_types)
        if g_type in type_to_idx:
            one_hot[type_to_idx[g_type]] = 1
        else:
            # If type is not in the list (e.g. OUTPUT alias), it gets all zeros for type part
            pass
            
        # Part 2: Dynamic features [expected_value, observed_value, is_diff]
        # Default initialization: 0 if simulation_data is None
        # If simulation_data is provided but node is missing: -1 (unknown)
        
        dynamic_feat = [0, 0, 0] # Default if simulation_data is None
        
        if simulation_data is not None:
            if name in simulation_data:
                data_point = simulation_data[name]
                
                exp_val = -1
                obs_val = -1
                
                # Handle different formats of simulation_data values
                if isinstance(data_point, dict):
                    exp_val = data_point.get('expected', -1)
                    obs_val = data_point.get('observed', -1)
                elif isinstance(data_point, (list, tuple)) and len(data_point) >= 2:
                    exp_val = data_point[0]
                    obs_val = data_point[1]
                elif isinstance(data_point, (int, float)):
                    # If single value, assume it's observed
                    obs_val = data_point
                    
                # Calculate is_diff
                # is_diff is 1 if both are known (not -1) and different
                if exp_val != -1 and obs_val != -1:
                    is_diff = 1 if exp_val != obs_val else 0
                else:
                    is_diff = 0 # Cannot determine difference if one is unknown
                
                dynamic_feat = [exp_val, obs_val, is_diff]
            else:
                # Node not in simulation data (e.g. internal node not probed)
                dynamic_feat = [-1, -1, 0]
        
        x_features.append(one_hot + dynamic_feat)

    x = torch.tensor(x_features, dtype=torch.float)
    
    data = Data(x=x, edge_index=edge_index)
    # Store node names to map back indices to circuit nodes
    data.node_names = node_names 
    
    return data

if __name__ == "__main__":
    # Create a dummy .bench file content
    dummy_bench_content = """
# c17 ISCAS85 benchmark (simplified)
INPUT(1)
INPUT(2)
INPUT(3)
INPUT(6)
INPUT(7)
10 = NAND(1, 3)
11 = NAND(3, 6)
16 = NAND(2, 11)
19 = NAND(11, 7)
22 = NAND(10, 16)
23 = NAND(16, 19)
"""
    dummy_filename = "dummy_c17.bench"
    try:
        with open(dummy_filename, "w") as f:
            f.write(dummy_bench_content)
        
        print(f"Created temporary file: {dummy_filename}")
        
        # 1. Parse
        print("Parsing .bench file...")
        G = parse_bench_to_networkx(dummy_filename)
        if G:
            print(f"Parsed Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
            
            # 2. Convert to PyG
            print("Converting to PyG Data object...")
            
            # Create dummy simulation data
            # Format: {node_name: {'expected': val, 'observed': val}}
            # Let's simulate a fault at node 22 where it is stuck-at-0 (expected 1, observed 0)
            sim_data = {
                '1': {'expected': 1, 'observed': 1},
                '2': {'expected': 0, 'observed': 0},
                '3': {'expected': 1, 'observed': 1},
                '6': {'expected': 0, 'observed': 0},
                '7': {'expected': 1, 'observed': 1},
                '10': {'expected': 0, 'observed': 0}, # NAND(1,1)=0
                '11': {'expected': 1, 'observed': 1}, # NAND(1,0)=1
                '16': {'expected': 1, 'observed': 1}, # NAND(0,1)=1
                '19': {'expected': 0, 'observed': 0}, # NAND(1,1)=0
                '22': {'expected': 1, 'observed': 0}, # NAND(0,1)=1. FAULT!
                '23': {'expected': 1, 'observed': 1}  # NAND(1,0)=1
            }
            
            data = convert_to_pyg_data(G, simulation_data=sim_data)
            
            print("\nGenerated PyG Data Object:")
            print(data)
            print(f"Node Features (x) shape: {data.x.shape}")
            print(f"Edge Index shape: {data.edge_index.shape}")
            
            # Verification
            # Features: 8 types + 3 dynamic = 11
            print(f"Feature dimension check: {'Pass' if data.x.shape[1] == 11 else 'Fail'}")
            
            # Check specific node feature (Node 22)
            # Find index of '22'
            idx_22 = data.node_names.index('22')
            feat_22 = data.x[idx_22]
            print(f"Features for Node '22': {feat_22.tolist()}")
            # Expected dynamic part (last 3): [1.0, 0.0, 1.0]
            print(f"Dynamic features for '22': {feat_22[-3:].tolist()}")
            
    finally:
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)
            print(f"Removed temporary file: {dummy_filename}")
