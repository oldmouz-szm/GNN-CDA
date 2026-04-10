import sys
import os
import time
import random
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from circuit_gnn_converter import parse_bench_to_networkx, convert_to_pyg_data
from circuit_simulator import CircuitSimulator
from train_gnn import FaultDiagnosisGNN, train
from generate_dataset import generate_dataset
from torch_geometric.loader import DataLoader
import torch.nn as nn

def set_seed(seed):
    random.seed(seed)
    np.random.seed.seed(seed)
    torch.manual_seed(seed)

def train_model(G, num_samples, fault_counts, fault_probs, num_epochs, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    num_node_features = 11
    model = FaultDiagnosisGNN(num_node_features=num_node_features)
    
    dataset = generate_dataset(
        G,
        num_samples=num_samples,
        fault_cardinalities=fault_counts,
        fault_cardinality_probs=fault_probs,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=True)
    
    if dataset[0].num_node_features != num_node_features:
        model = FaultDiagnosisGNN(num_node_features=dataset[0].num_node_features)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()
    
    losses = []
    model.train()
    for epoch in range(num_epochs):
        loss = train(model, loader, optimizer, criterion, 'cpu')
        losses.append(loss)
    
    return model, losses

def evaluate_model(model, simulator, G, num_cases=50, num_faults=3, timeout_seconds=10, seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    searcher_like = type('obj', (object,), {
        'simulator': simulator,
        'gnn_model': model,
        'node_names': sorted(list(G.nodes())),
        'device': 'cpu',
    })()
    
    node_names = sorted(list(G.nodes()))
    output_nodes = sorted([n for n in G.nodes() if G.out_degree(n) == 0],
                          key=lambda x: (0, int(x)) if str(x).isdigit() else (1, x))
    
    successes = 0
    total_steps = 0
    total_cards = 0
    
    for _ in range(num_cases):
        input_vec = {n: random.randint(0, 1) for n in simulator.input_nodes}
        
        current_injection = []
        for _ in range(num_faults):
            while True:
                f_cand = (random.choice(node_names), random.randint(0, 1))
                if not any(e[0] == f_cand[0] for e in current_injection):
                    current_injection.append(f_cand)
                    break
        current_injection.sort(key=lambda x: x[0])
        
        observed = simulator.simulate(input_vec, fault=current_injection)
        expected = simulator.simulate(input_vec, fault=None)
        
        sim_data = {}
        for node in node_names:
            sim_data[node] = {
                'expected': expected.get(node, 0),
                'observed': observed.get(node, 0)
            }
        data = convert_to_pyg_data(G, simulation_data=sim_data)
        
        with torch.no_grad():
            batch = torch.zeros(data.num_nodes, dtype=torch.long)
            scores = model(data.x, data.edge_index, batch).squeeze()
        score_map = {name: s.item() for name, s in zip(node_names, scores)}
        
        import heapq
        open_set = []
        heapq.heappush(open_set, (0.0, 0, tuple()))
        visited = set()
        steps = 0
        found = False
        card = 0
        
        start_t = time.time()
        while open_set and (time.time() - start_t) < timeout_seconds:
            cost, num_f, current_faults = heapq.heappop(open_set)
            
            if current_faults in visited:
                continue
            visited.add(current_faults)
            steps += 1
            
            sim_vals = simulator.simulate(input_vec, fault=list(current_faults))
            mismatches = [n for n, obs_val in observed.items() if n in sim_vals and sim_vals[n] != obs_val]
            
            if not mismatches:
                found = True
                card = len(current_faults)
                break
            
            if num_f >= num_faults:
                continue
            
            candidates = sorted(mismatches, key=lambda n: -score_map.get(n, 0.5))
            for cand in candidates[:3]:
                for val in [0, 1]:
                    new_faults = tuple(sorted(current_faults + ((cand, val),)))
                    if new_faults not in visited:
                        step_cost = -torch.log(torch.tensor(score_map.get(cand, 0.5) + 1e-8)).item()
                        new_cost = cost + step_cost
                        heapq.heappush(open_set, (new_cost, len(new_faults), new_faults))
        
        if found:
            successes += 1
            total_steps += steps
            total_cards += card
        else:
            total_steps += steps
    
    success_rate = successes / num_cases * 100
    avg_steps = total_steps / num_cases if num_cases > 0 else 0
    avg_card = total_cards / successes if successes > 0 else 0
    
    return success_rate, avg_steps, avg_card

def run_sweep():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    
    configs = [
        {
            'tier': 'Tiny',
            'name': 'b01_C',
            'bench': os.path.join(base_dir, 'ITC99', 'b01_C.bench'),
            'num_samples': 500,
            'fault_counts': [1, 2, 5, 10],
            'fault_probs': [0.4, 0.3, 0.2, 0.1],
            'epoch_range': [5, 10, 15, 20, 25, 30, 40, 50],
            'num_faults': 2,
            'num_cases': 30,
        },
        {
            'tier': 'Small',
            'name': 'c432',
            'bench': os.path.join(base_dir, 'iscas85', 'bench', 'c432.bench'),
            'num_samples': 1000,
            'fault_counts': [1, 2, 5, 10],
            'fault_probs': [0.35, 0.3, 0.2, 0.15],
            'epoch_range': [5, 10, 15, 20, 25, 30, 40, 50],
            'num_faults': 3,
            'num_cases': 30,
        },
        {
            'tier': 'Medium',
            'name': 'c1355',
            'bench': os.path.join(base_dir, 'iscas85', 'bench', 'c1355.bench'),
            'num_samples': 2000,
            'fault_counts': [1, 2, 5, 10, 20],
            'fault_probs': [0.3, 0.25, 0.2, 0.15, 0.1],
            'epoch_range': [5, 10, 15, 20, 25, 30, 40, 50],
            'num_faults': 3,
            'num_cases': 30,
        },
        {
            'tier': 'Large',
            'name': 'c6288',
            'bench': os.path.join(base_dir, 'iscas85', 'bench', 'c6288.bench'),
            'num_samples': 5000,
            'fault_counts': [1, 2, 5, 10, 20, 50],
            'fault_probs': [0.25, 0.2, 0.2, 0.15, 0.12, 0.08],
            'epoch_range': [10, 15, 20, 25, 30, 40, 50, 60],
            'num_faults': 5,
            'num_cases': 20,
        },
    ]
    
    results = {}
    
    for cfg in configs:
        tier = cfg['tier']
        name = cfg['name']
        print(f"\n{'='*70}")
        print(f"  TIER: {tier} | Circuit: {name}")
        print(f"{'='*70}")
        
        G = parse_bench_to_networkx(cfg['bench'])
        simulator = CircuitSimulator(G)
        num_nodes = len(G.nodes())
        print(f"  Nodes: {num_nodes}, Samples: {cfg['num_samples']}, Test faults: {cfg['num_faults']}")
        
        tier_results = []
        
        for num_epochs in cfg['epoch_range']:
            print(f"\n  --- Epochs={num_epochs} ---")
            
            t0 = time.time()
            model, losses = train_model(
                G, cfg['num_samples'], cfg['fault_counts'], cfg['fault_probs'], num_epochs
            )
            train_time = time.time() - t0
            
            final_loss = losses[-1]
            min_loss = min(losses)
            min_loss_epoch = losses.index(min_loss) + 1
            
            success_rate, avg_steps, avg_card = evaluate_model(
                model, simulator, G,
                num_cases=cfg['num_cases'],
                num_faults=cfg['num_faults'],
                timeout_seconds=10,
            )
            
            print(f"    Loss: final={final_loss:.4f}, min={min_loss:.4f}@epoch{min_loss_epoch}")
            print(f"    Success: {success_rate:.1f}%, AvgSteps: {avg_steps:.1f}, AvgCard: {avg_card:.2f}, TrainTime: {train_time:.1f}s")
            
            tier_results.append({
                'epochs': num_epochs,
                'final_loss': final_loss,
                'min_loss': min_loss,
                'min_loss_epoch': min_loss_epoch,
                'success_rate': success_rate,
                'avg_steps': avg_steps,
                'avg_card': avg_card,
                'train_time': train_time,
            })
        
        results[tier] = tier_results
        
        print(f"\n  === {tier} Summary ===")
        print(f"  {'Epochs':<8} {'Loss':<10} {'MinLoss@Ep':<12} {'Success%':<10} {'AvgSteps':<10} {'AvgCard':<10} {'Time(s)':<8}")
        print(f"  {'-'*68}")
        for r in tier_results:
            print(f"  {r['epochs']:<8} {r['final_loss']:<10.4f} {r['min_loss']:.4f}@{r['min_loss_epoch']:<5} {r['success_rate']:<10.1f} {r['avg_steps']:<10.1f} {r['avg_card']:<10.2f} {r['train_time']:<8.1f}")
        
        best = max(tier_results, key=lambda x: (x['success_rate'], -x['avg_steps']))
        print(f"  >>> Best: epochs={best['epochs']}, success={best['success_rate']:.1f}%, steps={best['avg_steps']:.1f}")
    
    print(f"\n{'='*70}")
    print(f"  FINAL RECOMMENDATIONS")
    print(f"{'='*70}")
    for tier, tier_results in results.items():
        best = max(tier_results, key=lambda x: (x['success_rate'], -x['avg_steps']))
        print(f"  {tier:8s}: recommended epochs = {best['epochs']:3d} (success={best['success_rate']:.1f}%, avg_steps={best['avg_steps']:.1f})")

if __name__ == "__main__":
    run_sweep()
