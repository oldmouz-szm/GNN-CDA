import torch
import random
import networkx as nx
from circuit_simulator import CircuitSimulator
from circuit_gnn_converter import parse_bench_to_networkx, convert_to_pyg_data

def generate_dataset(nx_graph, num_samples=1000, fault_cardinalities=None, fault_cardinality_probs=None):
    """
    Generates a dataset for circuit diagnosis.
    
    Args:
        nx_graph (nx.DiGraph): The circuit graph.
        num_samples (int): Number of samples to generate.
        fault_cardinalities (list[int] | None): Candidate fault counts per sample,
            e.g. [1, 2, 5, 10]. Defaults to [1, 2].
        fault_cardinality_probs (list[float] | None): Optional probabilities matching
            fault_cardinalities. If None, defaults to [0.7, 0.3] for [1, 2],
            otherwise uses uniform probabilities.
        
    Returns:
        list: List of PyG Data objects.
    """
    simulator = CircuitSimulator(nx_graph)
    dataset = []
    
    # Get node list for mapping labels
    node_names = sorted(list(nx_graph.nodes()))
    node_map = {name: i for i, name in enumerate(node_names)}
    
    # Identify internal nodes (candidates for fault injection)
    # We can include INPUTs as faulty too, but usually we care about gates.
    # Let's include all nodes as potential fault locations.
    candidate_fault_nodes = node_names

    if fault_cardinalities is None:
        fault_cardinalities = [1, 2]
    fault_cardinalities = [int(k) for k in fault_cardinalities if int(k) > 0]
    if not fault_cardinalities:
        fault_cardinalities = [1]

    if fault_cardinality_probs is None:
        if fault_cardinalities == [1, 2]:
            fault_cardinality_probs = [0.7, 0.3]
        else:
            fault_cardinality_probs = [1.0 / len(fault_cardinalities)] * len(fault_cardinalities)
    elif len(fault_cardinality_probs) != len(fault_cardinalities):
        raise ValueError("fault_cardinality_probs length must match fault_cardinalities")
    
    for _ in range(num_samples):
        # 1. Random Inputs
        input_values = {n: random.randint(0, 1) for n in simulator.input_nodes}
        
        # 2. Healthy Run (Expected Values)
        expected_values = simulator.simulate(input_values, fault=None)
        
        # 3. Faulty Run (Observed Values)
        requested_k = random.choices(fault_cardinalities, weights=fault_cardinality_probs, k=1)[0]
        k = min(max(1, requested_k), len(candidate_fault_nodes))
        selected_nodes = random.sample(candidate_fault_nodes, k)
        fault_pairs = [(n, random.randint(0, 1)) for n in selected_nodes]

        fault_obj = fault_pairs[0] if k == 1 else fault_pairs
        fault_nodes_set = set(selected_nodes)
        
        observed_values = simulator.simulate(input_values, fault=fault_obj)
        
        # 4. Data Packaging
        # Prepare simulation data for converter
        sim_data = {}
        for node in node_names:
            sim_data[node] = {
                'expected': expected_values.get(node, 0),
                'observed': observed_values.get(node, 0)
            }
            
        # Create PyG Data object
        data = convert_to_pyg_data(nx_graph, simulation_data=sim_data)
        
        # Add Label (Multi-label classification: 1 if faulty, 0 otherwise)
        y = torch.zeros(data.num_nodes, dtype=torch.float)
        for i, name in enumerate(node_names):
            if name in fault_nodes_set:
                y[i] = 1.0
        data.y = y
        
        # Add metadata (optional but helpful)
        data.fault_info = str(fault_obj)
        
        dataset.append(data)
        
    return dataset

if __name__ == "__main__":
    # Verification Block
    print("--- Verifying CircuitSimulator and Data Generation ---")
    
    # 1. Create a dummy .bench file
    dummy_bench_content = """
# c17 dummy
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
    dummy_filename = "verify_c17.bench"
    with open(dummy_filename, "w") as f:
        f.write(dummy_bench_content)
        
    try:
        # 2. Parse
        print(f"Parsing {dummy_filename}...")
        G = parse_bench_to_networkx(dummy_filename)
        
        # 3. Generate small dataset
        print("Generating 10 samples...")
        dataset = generate_dataset(G, num_samples=10)
        
        # 4. Verify first sample
        sample = dataset[0]
        print("\n--- Sample 1 Verification ---")
        print(f"Fault Info: {sample.fault_info}")
        print(f"Label (y): {sample.y.item()}")
        
        # Check features of the faulty node
        faulty_node_idx = sample.y.item()
        faulty_node_name = sample.node_names[faulty_node_idx]
        print(f"Faulty Node Name: {faulty_node_name}")
        
        features = sample.x[faulty_node_idx]
        # Features: [8 types, Expected, Observed, IsDiff]
        expected = features[-3].item()
        observed = features[-2].item()
        is_diff = features[-1].item()
        
        print(f"Features for faulty node:")
        print(f"  Expected: {expected}")
        print(f"  Observed: {observed}")
        print(f"  Is Diff: {is_diff}")
        
        # Verify IsDiff logic
        calc_diff = 1.0 if expected != observed else 0.0
        if is_diff == calc_diff:
            print("SUCCESS: 'is_diff' feature is correctly calculated.")
        else:
            print(f"FAIL: 'is_diff' mismatch. Feature: {is_diff}, Calculated: {calc_diff}")
            
    finally:
        import os
        if os.path.exists(dummy_filename):
            os.remove(dummy_filename)
