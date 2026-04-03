import argparse
import csv
import random
import statistics
import time
from typing import Dict, List, Optional, Set, Tuple

from circuit_gnn_converter import parse_bench_to_networkx
from circuit_simulator import CircuitSimulator
from safari_exact import SafariMBD

Diagnosis = Set[Tuple[str, int]]


def get_primary_inputs(graph):
    return [n for n, d in graph.nodes(data=True) if d.get("type") == "INPUT"]


def get_primary_outputs(graph):
    return [n for n in graph.nodes() if graph.out_degree(n) == 0]


def get_components(graph):
    # In this repository, assumable components are non-input gates.
    return [n for n, d in graph.nodes(data=True) if d.get("type") != "INPUT"]


def smallest_cardinality_diagnosis(diagnoses) -> Optional[Diagnosis]:
    if not diagnoses:
        return None
    best = None
    best_card = 10**9
    for d in diagnoses:
        card = len(d)
        if card < best_card:
            best_card = card
            best = set(d)
    return best


def run_safari_once(simulator, inputs: Dict[str, int], outputs: Dict[str, int], m: int, n: int):
    safari = SafariMBD(simulator, max_tries=n, max_climb=m)
    t0 = time.time()
    diagnoses = safari.run(inputs, outputs)
    dt = time.time() - t0
    best = smallest_cardinality_diagnosis(diagnoses)
    return diagnoses, best, dt


def make_alphas(simulator, in_nodes: List[str], out_nodes: List[str], comps: List[str],
                n_for_generation: int, k_max: int, seed: int):
    """
    Reproduce Algorithm 3 in the paper:
    - For observation generation, use Safari with M=|COMPS| and N=20.
    - Greedily flip each output bit and keep observations that increase
      the smallest diagnosis cardinality.
    Returns a list of tuples: (inputs, flipped_outputs, best_cardinality)
    """
    random.seed(seed)

    observations = []
    seen = set()

    m_generation = len(comps)

    for _ in range(k_max):
        alpha_inputs = {node: random.randint(0, 1) for node in in_nodes}
        nominal_state = simulator.simulate(alpha_inputs, fault=None)
        beta_outputs = {node: nominal_state[node] for node in out_nodes}

        c_best = 0
        for out_node in out_nodes:
            alpha_n_outputs = dict(beta_outputs)
            alpha_n_outputs[out_node] = 1 - alpha_n_outputs[out_node]

            _, best_diag, _ = run_safari_once(
                simulator,
                alpha_inputs,
                alpha_n_outputs,
                m=m_generation,
                n=n_for_generation,
            )

            if best_diag is None:
                continue

            diag_card = len(best_diag)
            if diag_card > c_best:
                c_best = diag_card
                key = (
                    tuple(sorted(alpha_inputs.items())),
                    tuple(sorted(alpha_n_outputs.items())),
                )
                if key not in seen:
                    seen.add(key)
                    observations.append((dict(alpha_inputs), dict(alpha_n_outputs), diag_card))

    return observations


def benchmark_observations(simulator, observations, m_default: int, n_default: int,
                           repeats: int, seed: int):
    random.seed(seed)
    rows = []

    for idx, (inputs, outputs, target_card) in enumerate(observations, start=1):
        runtimes = []
        best_cards = []
        diagnosis_counts = []

        for rep in range(repeats):
            # Keep reproducibility while preserving stochastic behavior per repeat.
            random.seed(seed + idx * 10000 + rep)
            diagnoses, best_diag, dt = run_safari_once(
                simulator,
                inputs,
                outputs,
                m=m_default,
                n=n_default,
            )
            runtimes.append(dt)
            diagnosis_counts.append(len(diagnoses))
            best_cards.append(len(best_diag) if best_diag is not None else -1)

        row = {
            "obs_id": idx,
            "target_cardinality_from_alg3": target_card,
            "runtime_min_s": min(runtimes),
            "runtime_avg_s": statistics.mean(runtimes),
            "runtime_max_s": max(runtimes),
            "best_diag_card_min": min(best_cards),
            "best_diag_card_avg": statistics.mean(best_cards),
            "best_diag_card_max": max(best_cards),
            "diag_count_avg": statistics.mean(diagnosis_counts),
        }
        rows.append(row)

    return rows


def save_csv(path: str, rows):
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Reproduce SAFARI paper-style experiment")
    parser.add_argument("--circuit", type=str, required=True, help="Circuit name, e.g. c17, c432")
    parser.add_argument("--k", type=int, default=10, help="Algorithm 3 K (max observation generations)")
    parser.add_argument("--gen_n", type=int, default=20, help="Algorithm 3 generation N (paper uses 20)")
    parser.add_argument("--eval_n", type=int, default=4, help="Default evaluation N (paper default 4)")
    parser.add_argument("--eval_m", type=int, default=8, help="Default evaluation M (paper default 8)")
    parser.add_argument("--repeats", type=int, default=10, help="Repeated runs per observation (paper uses 10)")
    parser.add_argument("--seed", type=int, default=20260313, help="Random seed")
    parser.add_argument("--csv", type=str, default="safari_paper_results.csv", help="Output CSV path")
    args = parser.parse_args()

    bench_file = f"iscas85/bench/{args.circuit}.bench"
    graph = parse_bench_to_networkx(bench_file)
    if graph is None:
        raise SystemExit(f"Failed to parse bench file: {bench_file}")

    simulator = CircuitSimulator(graph)
    in_nodes = get_primary_inputs(graph)
    out_nodes = get_primary_outputs(graph)
    comps = get_components(graph)

    print(f"Circuit: {args.circuit}")
    print(f"Inputs: {len(in_nodes)}, Outputs: {len(out_nodes)}, Components: {len(comps)}")
    print("Phase 1/2: Generate observations with Algorithm 3 settings...")
    print(f"  generation params: M=|COMPS|={len(comps)}, N={args.gen_n}, K={args.k}")

    observations = make_alphas(
        simulator,
        in_nodes,
        out_nodes,
        comps,
        n_for_generation=args.gen_n,
        k_max=args.k,
        seed=args.seed,
    )

    print(f"Generated observations: {len(observations)}")
    if not observations:
        print("No observations generated. Try increasing --k.")
        return

    print("Phase 2/2: Evaluate with paper default configuration...")
    print(f"  evaluation params: M={args.eval_m}, N={args.eval_n}, repeats={args.repeats}")

    rows = benchmark_observations(
        simulator,
        observations,
        m_default=args.eval_m,
        n_default=args.eval_n,
        repeats=args.repeats,
        seed=args.seed,
    )

    save_csv(args.csv, rows)

    overall_avg = statistics.mean([r["runtime_avg_s"] for r in rows])
    print(f"Saved CSV: {args.csv}")
    print(f"Observation count: {len(rows)}")
    print(f"Overall avg runtime per observation (avg across repeats): {overall_avg:.6f}s")


if __name__ == "__main__":
    main()
