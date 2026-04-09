import torch
import random
import time
import os
import heapq
import math
import numpy as np
from circuit_gnn_converter import convert_to_pyg_data
from circuit_simulator import CircuitSimulator
import networkx as nx


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

class GNNCdaDiagnosis:
    def __init__(self, simulator, gnn_model, device='cpu'):
        self.simulator = simulator
        self.gnn_model = gnn_model
        self.device = device
        self.gnn_model.eval()
        self.node_names = sorted(list(simulator.G.nodes()))

        # Cache for ancestors to speed up CDA
        self.output_ancestors = {}

    def get_ancestors(self, node):
        if node not in self.output_ancestors:
            if node in self.simulator.G:
                ancs = nx.ancestors(self.simulator.G, node)
                ancs.add(node)
                self.output_ancestors[node] = ancs
            else:
                self.output_ancestors[node] = set()
        return self.output_ancestors[node]

    def diagnose(self, input_values, observed_values, max_faults=20, max_steps=1000, use_gnn=True, max_time_seconds=180):
        """
        Performs GNN-Guided Conflict-Directed A* Search.
        """
        # 1. Get GNN Priors
        expected_values = self.simulator.simulate(input_values, fault=None)
        
        if use_gnn:
            sim_data = {}
            for node in self.node_names:
                sim_data[node] = {
                    'expected': expected_values.get(node, 0),
                    'observed': observed_values.get(node, 0)
                }
                
            data = convert_to_pyg_data(self.simulator.G, simulation_data=sim_data)
            data = data.to(self.device)
            
            with torch.no_grad():
                batch = torch.zeros(data.num_nodes, dtype=torch.long, device=self.device)
                scores = self.gnn_model(data.x, data.edge_index, batch).squeeze()
                
            score_map = {name: s.item() for name, s in zip(self.node_names, scores)}
        else:
            score_map = {name: 0.5 for name in self.node_names} # Uniform probability

        search_start_time = time.time()
        search_deadline = None
        if max_time_seconds is not None and max_time_seconds > 0:
            search_deadline = search_start_time + max_time_seconds

        def run_search(active_score_map, deadline, branch_limit):
            # Priority Queue: (cost, num_faults, fault_set_tuple)
            # cost = sum(-log(P(node))) for nodes in fault_set
            open_set = []
            heapq.heappush(open_set, (0.0, 0, tuple()))

            visited = set()
            steps = 0

            while open_set:
                if deadline is not None and time.time() >= deadline:
                    return None, steps
                if max_steps is not None and steps >= max_steps:
                    return None, steps

                cost, _, current_faults = heapq.heappop(open_set)

                if current_faults in visited:
                    continue
                visited.add(current_faults)
                steps += 1

                # Simulate
                sim_vals = self.simulator.simulate(input_values, fault=list(current_faults))

                # Check Consistency & Identify Mismatches
                mismatches = []
                for node, obs_val in observed_values.items():
                    if node in sim_vals and sim_vals[node] != obs_val:
                        mismatches.append(node)

                if not mismatches:
                    return list(current_faults), steps

                # Pruning: Max faults reached
                if len(current_faults) >= max_faults:
                    continue

                # CDA Step: Pick a mismatch and expand its ancestors
                # Heuristic: Pick mismatch with smallest cone (fewest candidates) to minimize branching
                best_mismatch = min(mismatches, key=lambda n: len(self.get_ancestors(n)))
                candidates = self.get_ancestors(best_mismatch)

                # Filter candidates: must not be already in current_faults
                existing_fault_nodes = {f[0] for f in current_faults}
                valid_candidates = [n for n in candidates if n not in existing_fault_nodes]

                # Sort by score descending (Try most likely first)
                valid_candidates.sort(key=lambda n: active_score_map.get(n, 0), reverse=True)

                for node in valid_candidates[:branch_limit]:
                    node_score = active_score_map.get(node, 1e-6)
                    if node_score < 1e-6:
                        node_score = 1e-6

                    # Cost update: Add -log(P) of the new fault
                    step_cost = -math.log(node_score)

                    for val in [0, 1]:
                        new_fault = (node, val)
                        # Sort to ensure uniqueness of tuple
                        new_faults = tuple(sorted(current_faults + (new_fault,)))

                        if new_faults not in visited:
                            new_total_cost = cost + step_cost
                            heapq.heappush(open_set, (new_total_cost, len(new_faults), new_faults))

            return None, steps

        # 2. A* Search Execution (time-bound only)
        return run_search(score_map, search_deadline, 20)

def evaluate_astar_performance(circuit_name, bench_path, gnn_model, num_test_samples=20, num_injected_faults=2, timeout_seconds=180, gnn_only=False):
    from circuit_gnn_converter import parse_bench_to_networkx
    
    print(f"\nEvaluating GNN-CDA Performance on {circuit_name} ({num_injected_faults} Faults)...")
    
    G = parse_bench_to_networkx(bench_path)
    simulator = CircuitSimulator(G)
    searcher = GNNCdaDiagnosis(simulator, gnn_model)
    node_names = sorted(list(G.nodes()))

    def _node_sort_key(name):
        s = str(name)
        return (0, int(s)) if s.isdigit() else (1, s)

    output_nodes = sorted([n for n in G.nodes() if G.out_degree(n) == 0], key=_node_sort_key)
    input_nodes = sorted(simulator.input_nodes, key=_node_sort_key)
    obs_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "obs")
    os.makedirs(obs_dir, exist_ok=True)

    # Cross-project bridge artifacts for OOE-style experiments.
    ooe_obs_path = os.path.join(obs_dir, f"{circuit_name}_iscas85.obs")
    ooe_map_path = os.path.join(obs_dir, f"{circuit_name}_io_map.txt")
    with open(ooe_map_path, "w", encoding="utf-8") as f:
        f.write("# Auto-generated I/O mapping for OOE comparison\n")
        f.write("# i-index to real input node\n")
        for i, node in enumerate(input_nodes, start=1):
            f.write(f"i{i}={node}\n")
        f.write("# o-index to real output node\n")
        for i, node in enumerate(output_nodes, start=1):
            f.write(f"o{i}={node}\n")

    # Rewrite the file on each run so observation indices always match this run's cases.
    with open(ooe_obs_path, "w", encoding="utf-8") as f:
        f.write("")
    
    # Generate Test Cases
    test_cases = []
    for case_idx in range(num_test_samples):
        input_vec = {n: random.randint(0, 1) for n in simulator.input_nodes}
        
        # Inject N Faults
        current_injection = []
        for _ in range(num_injected_faults):
            # Ensure unique nodes for faults
            while True:
                f_cand = (random.choice(node_names), random.randint(0, 1))
                if not any(existing[0] == f_cand[0] for existing in current_injection):
                    current_injection.append(f_cand)
                    break
        
        # Sort for consistency
        current_injection.sort(key=lambda x: x[0])
        true_faults = current_injection
        observed = simulator.simulate(input_vec, fault=true_faults)

        # Save input/output observations in obs/*.txt using key=value lines.
        obs_path = os.path.join(obs_dir, f"{circuit_name}_obs_{case_idx + 1}.txt")
        with open(obs_path, "w", encoding="utf-8") as f:
            f.write("# Auto-generated observation\n")
            f.write(f"# faults={','.join(node for node, _ in true_faults)}\n")
            f.write("# inputs\n")
            for node in sorted(input_vec.keys(), key=_node_sort_key):
                f.write(f"{node}={input_vec[node]}\n")
            f.write("# outputs\n")
            for node in output_nodes:
                f.write(f"{node}={observed.get(node, 0)}\n")

        # Save OOE-style observation line: (circuit,idx,[i1,-i2,...,o1,-o2,...]).
        io_literals = []
        for i, node in enumerate(input_nodes, start=1):
            val = input_vec.get(node, 0)
            io_literals.append(f"i{i}" if val == 1 else f"-i{i}")
        for i, node in enumerate(output_nodes, start=1):
            val = observed.get(node, 0)
            io_literals.append(f"o{i}" if val == 1 else f"-o{i}")

        with open(ooe_obs_path, "a", encoding="utf-8") as f:
            f.write(f"({circuit_name},{case_idx + 1},[{','.join(io_literals)}]).\n")
        
        test_cases.append({
            'input': input_vec,
            'observed': observed,
            'true_faults': true_faults
        })

    def fault_hit_rate(pred_faults, true_faults):
        # Hit rate is undefined when diagnosis set is empty; return None and exclude from averages.
        if pred_faults is None:
            return None
        pred_set = set(pred_faults)
        if not pred_set:
            return None
        true_set = set(true_faults)
        if not true_set:
            return 0.0
        return len(true_set.intersection(pred_set)) / len(pred_set)
        
    if gnn_only:
        header = (
            f"{'Case':<5} | {'True Faults':<25} | {'GNN Steps':<10} | {'GNN Card':<9} | {'GNN Time':<9} | {'GNN Res':<8} | {'HitRate':<8} | {'Pred Faults':<25}"
        )
    else:
        header = (
            f"{'Case':<5} | {'True Faults':<25} | {'GNN Steps':<10} | {'GNN Card':<9} | {'GNN Time':<9} | "
            f"{'Base Steps':<10} | {'Base Card':<9} | {'Base Time':<9} | {'GNN Res':<8} | {'Base Res':<8} | {'HitRate':<8} | {'Pred Faults':<25}"
        )
    print(header)
    print("-" * len(header))
    
    gnn_total_steps = 0
    base_total_steps = 0
    gnn_total_time = 0.0
    base_total_time = 0.0
    gnn_success = 0
    base_success = 0
    gnn_cards = []
    base_cards = []
    hit_rates = []
    
    for i, case in enumerate(test_cases, start=1):
        # Run GNN-CDA
        start_t = time.time()
        pred_gnn, steps_gnn = searcher.diagnose(
            case['input'], 
            case['observed'], 
            max_faults=num_injected_faults, # Allow up to the number of injected faults
            use_gnn=True,
            max_time_seconds=timeout_seconds,
        )
        t_gnn = time.time() - start_t
        
        pred_base = None
        steps_base = 0
        t_base = 0.0
        if not gnn_only:
            # Run Baseline-CDA (No GNN guidance)
            start_t = time.time()
            pred_base, steps_base = searcher.diagnose(
                case['input'], 
                case['observed'], 
                max_faults=num_injected_faults, 
                use_gnn=False,
                max_time_seconds=timeout_seconds,
            )
            t_base = time.time() - start_t

        gnn_total_steps += steps_gnn
        gnn_total_time += t_gnn
        if not gnn_only:
            base_total_steps += steps_base
            base_total_time += t_base

        gnn_card = len(pred_gnn) if pred_gnn is not None else "NA"
        base_card = len(pred_base) if pred_base is not None else "NA"
        
        res_gnn = "SUCCESS" if pred_gnn is not None else "FAIL"
        res_base = "SUCCESS" if pred_base is not None else "FAIL"
        
        if pred_gnn is not None:
            gnn_success += 1
            gnn_cards.append(len(pred_gnn))
        if (not gnn_only) and pred_base is not None:
            base_success += 1
            base_cards.append(len(pred_base))
        
        true_str = str(case['true_faults'])
        selected_pred = pred_gnn if pred_gnn is not None else pred_base
        pred_str = str(selected_pred) if selected_pred is not None else "None"
        hit_rate = fault_hit_rate(selected_pred, case['true_faults'])
        if hit_rate is not None:
            hit_rates.append(hit_rate)
        
        # Truncate long strings just for display
        if len(true_str) > 24:
            true_str = true_str[:21] + "..."
        if len(pred_str) > 24:
            pred_str = pred_str[:21] + "..."
        hit_rate_str = f"{hit_rate:.2%}" if hit_rate is not None else "NA"
            
        if gnn_only:
            print(
                f"{i:<5} | {true_str:<25} | {steps_gnn:<10} | {gnn_card!s:<9} | {t_gnn:<8.3f}s | {res_gnn:<8} | {hit_rate_str:<8} | {pred_str:<25}"
            )
        else:
            print(
                f"{i:<5} | {true_str:<25} | {steps_gnn:<10} | {gnn_card!s:<9} | {t_gnn:<8.3f}s | "
                f"{steps_base:<10} | {base_card!s:<9} | {t_base:<8.3f}s | {res_gnn:<8} | {res_base:<8} | {hit_rate_str:<8} | {pred_str:<25}"
            )
        
    print("-" * len(header))
    gnn_avg_card = (sum(gnn_cards) / len(gnn_cards)) if gnn_cards else 0.0
    base_avg_card = (sum(base_cards) / len(base_cards)) if base_cards else 0.0
    avg_hit = (sum(hit_rates) / len(hit_rates)) if hit_rates else 0.0
    print(
        f"GNN-CDA  -> Avg Steps: {gnn_total_steps / num_test_samples:.2f}, "
        f"Avg Card: {gnn_avg_card:.2f}, Avg Hit: {avg_hit:.2%}, Avg Time: {gnn_total_time / num_test_samples:.3f}s, "
        f"Success: {gnn_success / num_test_samples:.1%}"
    )
    if not gnn_only:
        print(
            f"Base-CDA -> Avg Steps: {base_total_steps / num_test_samples:.2f}, "
            f"Avg Card: {base_avg_card:.2f}, Avg Time: {base_total_time / num_test_samples:.3f}s, "
            f"Success: {base_success / num_test_samples:.1%}"
        )

def main():
    # Integration Test
    import os
    import argparse
    from train_gnn import FaultDiagnosisGNN, train
    from generate_dataset import generate_dataset
    from torch_geometric.loader import DataLoader
    from circuit_gnn_converter import parse_bench_to_networkx
    
    parser = argparse.ArgumentParser(description='Run GNN-CDA Diagnosis on ISCAS-85 circuits')
    parser.add_argument('--circuit', type=str, default='c880', 
                        help='Name of the circuit (e.g., c432, c880, c1908)')
    parser.add_argument('--retrain', action='store_true', help='Force retraining of the model')
    parser.add_argument('--num_faults', type=int, default=2, help='Number of faults to inject and diagnose')
    parser.add_argument('--train_fault_counts', type=str, default='1,2', help='Comma-separated fault counts for training dataset, e.g. 1,2,5,10')
    parser.add_argument('--num_test_samples', type=int, default=500, help='Number of evaluation samples (cases)')
    parser.add_argument('--timeout_seconds', type=int, default=180, help='Timeout per diagnosis case in seconds')
    parser.add_argument('--seed', type=int, default=8, help='Fixed random seed for reproducible experiments')
    parser.add_argument('--eval_seed', type=int, default=None, help='Optional seed for evaluation cases; defaults to --seed')
    parser.add_argument('--bench_path', type=str, default=None,
                        help='Full path to .bench file. If not given, searches iscas85/bench and ITC99 directories.')
    parser.add_argument('--gnn_only', action='store_true', help='Only run GNN-CDA and skip Base-CDA evaluation')
    args = parser.parse_args()

    def pick_training_profile(num_nodes, requested_fault_counts):
        """Returns (num_samples, fault_counts, fault_probs, size_label)."""
        counts = requested_fault_counts
        probs = None

        if num_nodes > 10000:
            num_samples = 8000
            size_label = "Huge"
            if counts == [1, 2]:
                counts = [1, 2, 5, 10, 20, 50]
                probs = [0.25, 0.2, 0.2, 0.15, 0.12, 0.08]
        elif num_nodes > 2000:
            num_samples = 5000
            size_label = "Large"
            if counts == [1, 2]:
                counts = [1, 2, 5, 10, 20]
                probs = [0.3, 0.25, 0.2, 0.15, 0.1]
        elif num_nodes > 500:
            num_samples = 1000
            size_label = "Medium"
            if counts == [1, 2]:
                counts = [1, 2, 5, 10]
                probs = [0.35, 0.3, 0.2, 0.15]
        elif num_nodes > 50:
            num_samples = 500
            size_label = "Small"
            if counts == [1, 2]:
                counts = [1, 2, 3, 5]
                probs = [0.4, 0.3, 0.2, 0.1]
        else:
            num_samples = 200
            size_label = "Tiny"
            if counts == [1, 2]:
                counts = [1, 2, 3]
                probs = [0.5, 0.35, 0.15]

        return num_samples, counts, probs, size_label

    set_seed(args.seed)
    eval_seed = args.seed if args.eval_seed is None else args.eval_seed
    if eval_seed == args.seed:
        print(f"Seed: {args.seed}")
    else:
        print(f"Seed(train/eval): {args.seed}/{eval_seed}")

    try:
        train_fault_counts = [int(x.strip()) for x in args.train_fault_counts.split(',') if x.strip()]
        train_fault_counts = [k for k in train_fault_counts if k > 0]
        if not train_fault_counts:
            raise ValueError("empty")
    except Exception:
        raise ValueError(f"Invalid --train_fault_counts: {args.train_fault_counts}. Example: --train_fault_counts 1,2,5,10")
    
    circuit_name = args.circuit
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    if args.bench_path:
        bench_path = args.bench_path
    else:
        search_dirs = [
            os.path.join(base_dir, "iscas85", "bench"),
            os.path.join(base_dir, "ITC99"),
        ]
        bench_path = None
        for d in search_dirs:
            candidate = os.path.join(d, f"{circuit_name}.bench")
            if os.path.exists(candidate):
                bench_path = candidate
                break
        if bench_path is None:
            print(f"Error: Circuit file not found. Searched: {search_dirs}")
            exit(1)

    # Setup Model Path
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"{circuit_name}_gnn.pth")
    
    print(f"Targeting circuit: {circuit_name}")
    G = parse_bench_to_networkx(bench_path)
    
    # Determined from circuit_gnn_converter.py (8 types + 3 dynamic)
    num_node_features = 11 
    model = FaultDiagnosisGNN(num_node_features=num_node_features)
    
    # Check if we should load or train
    should_train = True
    if os.path.exists(model_path) and not args.retrain:
        print(f"Found saved model at {model_path}. Loading...")
        try:
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            should_train = False
            print("Model loaded successfully.")
        except Exception as e:
            print(f"Failed to load model: {e}. Proceeding to train.")
            should_train = True
            
    if should_train:
        print(f"Training model for {circuit_name}...")
        num_nodes = len(G.nodes())
        num_samples, train_fault_counts, train_fault_probs, size_label = pick_training_profile(
            num_nodes,
            train_fault_counts,
        )
        print(f"{size_label} circuit detected ({num_nodes} nodes). Using {num_samples} training samples.")
        print(f"Training fault counts: {train_fault_counts}")
        if train_fault_probs is not None:
            print(f"Training fault probs: {train_fault_probs}")
            
        dataset = generate_dataset(
            G,
            num_samples=num_samples,
            fault_cardinalities=train_fault_counts,
            fault_cardinality_probs=train_fault_probs,
        )
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Verify feature size matches our hardcoded assumption
        if dataset[0].num_node_features != num_node_features:
            print(f"Warning: Expected {num_node_features} features, got {dataset[0].num_node_features}.")
            # Re-initialize with correct size just in case
            model = FaultDiagnosisGNN(num_node_features=dataset[0].num_node_features)
        
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        criterion = torch.nn.BCELoss()
        
        model.train()
        for epoch in range(20):
            train(model, loader, optimizer, criterion, 'cpu')
            
        # Save the model
        torch.save(model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    # Keep evaluation case generation reproducible regardless of whether we trained or loaded.
    set_seed(eval_seed)

    evaluate_astar_performance(
        circuit_name,
        bench_path,
        model,
        num_test_samples=args.num_test_samples,
        num_injected_faults=args.num_faults,
        timeout_seconds=args.timeout_seconds,
        gnn_only=args.gnn_only,
    )

if __name__ == "__main__":
    main()
