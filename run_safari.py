import argparse
import random
import networkx as nx
import time
import statistics
from circuit_gnn_converter import parse_bench_to_networkx
from circuit_simulator import CircuitSimulator
from safari_exact import SafariMBD

def get_primary_outputs(G):
    return [n for n in G.nodes() if G.out_degree(n) == 0]

def get_primary_inputs(G):
    return [n for n, d in G.nodes(data=True) if d.get('type') == 'INPUT']

def main():
    parser = argparse.ArgumentParser(description="Run SAFARI Exact Diagnosis")
    parser.add_argument('--circuit', type=str, required=True, help="Circuit name (e.g., c17, c432)")
    parser.add_argument('--num_faults', type=int, default=1, help="Number of random faults to inject")
    parser.add_argument('--trials', type=int, default=1, help="Number of random trials")
    parser.add_argument('--m', type=int, default=10, help="SAFARI M parameter (Climb limit)")
    parser.add_argument('--n', type=int, default=5, help="SAFARI N parameter (Number of tries)")
    parser.add_argument('--sat_backend', type=str, default='auto', choices=['auto', 'pysat', 'dpll'],
                        help="SAT backend for RandomDiagnosis")
    parser.add_argument('--max_show', type=int, default=5, help="Max diagnosis candidates to print per trial")
    parser.add_argument('--summary_only', action='store_true',
                        help="Only print per-trial runtime/cardinality and final summary")
    
    args = parser.parse_args()
    
    bench_file = f"iscas85/bench/{args.circuit}.bench"
    print(f"Loading circuit from {bench_file}...")
    
    G = parse_bench_to_networkx(bench_file)
    if G is None:
        print("Failed to load circuit.")
        return
        
    simulator = CircuitSimulator(G)
    primary_inputs = get_primary_inputs(G)
    primary_outputs = get_primary_outputs(G)
    
    print(f"Components: {G.number_of_nodes()}, Inputs: {len(primary_inputs)}, Outputs: {len(primary_outputs)}")
    
    # Run trials
    trial_runtimes = []
    trial_min_cards = []
    trial_min_diag_counts = []
    start_time = time.time()
    
    for i in range(args.trials):
        print(f"\n--- Trial {i+1}/{args.trials} ---")
        
        # 1. Generate Random Inputs
        inputs = {node: random.randint(0, 1) for node in primary_inputs}
        
        # 2. Inject Random Faults (Ground Truth)
        # Avoid input nodes for faults (usually gates fail)
        possible_fault_nodes = [n for n in G.nodes() if G.nodes[n].get('type') != 'INPUT']
        if len(possible_fault_nodes) < args.num_faults:
            print("Not enough nodes for requested faults.")
            break
            
        fault_nodes = random.sample(possible_fault_nodes, args.num_faults)
        # Randomly assign s-a-0 or s-a-1
        ground_truth_fault = set()
        for fn in fault_nodes:
            val = random.randint(0, 1)
            ground_truth_fault.add((fn, val))
            
        if not args.summary_only:
            print(f"Ground Truth Fault: {ground_truth_fault}")
        
        # 3. Simulate to get Observations (The 'Error' Syndrome)
        # We need the output of the FAULTY circuit
        faulty_outputs = simulator.simulate(inputs, fault=ground_truth_fault)
        
        # Filter to only Primary Outputs (what we can actually observe)
        observations = {node: faulty_outputs[node] for node in primary_outputs}
        
        # 4. Run SAFARI
        safari = SafariMBD(simulator, max_tries=args.n, max_climb=args.m, sat_backend=args.sat_backend)
        print(f"Running SAFARI (N={args.n}, M={args.m}, SAT={args.sat_backend})...")
        
        t0 = time.time()
        diagnosis_results = safari.run(inputs, observations)
        trial_time = time.time() - t0
        trial_runtimes.append(trial_time)
        
        # 5. Check results
        if not args.summary_only:
            print(f"Found {len(diagnosis_results)} candidate diagnoses.")

        if diagnosis_results:
            min_card = min(len(d) for d in diagnosis_results)
            min_card_diags = [d for d in diagnosis_results if len(d) == min_card]
        else:
            min_card = -1
            min_card_diags = []

        trial_min_cards.append(min_card)
        trial_min_diag_counts.append(len(min_card_diags))

        print(f"  Trial Runtime: {trial_time:.4f}s")
        print(f"  Minimal Diagnosis Cardinality: {min_card}")
        print(f"  # Minimal-Cardinality Diagnoses: {len(min_card_diags)}")

        if not args.summary_only:
            shown = 0
            for d in min_card_diags:
                if shown < args.max_show:
                    print(f"  MinDiag(size={len(d)}): {d}")
                    shown += 1

            if len(min_card_diags) > args.max_show:
                print(f"  ... ({len(min_card_diags) - args.max_show} more min-card diagnoses omitted)")
            
    total_time = time.time() - start_time

    valid_cards = [c for c in trial_min_cards if c >= 0]
    print("\nSummary (Minimal Diagnosis + Runtime)")
    print(f"  Trials: {args.trials}")
    print(f"  Total Time: {total_time:.4f}s")

    if trial_runtimes:
        print(f"  Runtime Min/Avg/Max: {min(trial_runtimes):.4f}s / {statistics.mean(trial_runtimes):.4f}s / {max(trial_runtimes):.4f}s")

    if valid_cards:
        print(f"  Min-Card Min/Avg/Max: {min(valid_cards)} / {statistics.mean(valid_cards):.2f} / {max(valid_cards)}")
        print(f"  Avg #Min-Card Diagnoses: {statistics.mean(trial_min_diag_counts):.2f}")
    else:
        print("  No satisfiable diagnoses found in any trial.")

if __name__ == "__main__":
    main()