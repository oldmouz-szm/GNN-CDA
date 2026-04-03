import torch
import torch.nn.functional as F
from torch.nn import Linear, ReLU, Sigmoid, Sequential, BatchNorm1d
from torch_geometric.nn import GINConv
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
import numpy as np
import os

# Import our previous tools
from generate_dataset import generate_dataset
from circuit_gnn_converter import parse_bench_to_networkx

class FaultDiagnosisGNN(torch.nn.Module):
    def __init__(self, num_node_features):
        super(FaultDiagnosisGNN, self).__init__()
        
        # Use GIN (Graph Isomorphism Network) for better expressiveness (XOR support)
        dim = 64
        
        # MLP for GIN Layer 1
        nn1 = Sequential(Linear(num_node_features, dim), ReLU(), Linear(dim, dim))
        self.conv1 = GINConv(nn1)
        self.bn1 = BatchNorm1d(dim)
        
        # MLP for GIN Layer 2
        nn2 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv2 = GINConv(nn2)
        self.bn2 = BatchNorm1d(dim)
        
        # MLP for GIN Layer 3
        nn3 = Sequential(Linear(dim, dim), ReLU(), Linear(dim, dim))
        self.conv3 = GINConv(nn3)
        self.bn3 = BatchNorm1d(dim)
        
        # Classifier: Map node embeddings to a single score
        self.classifier = Sequential(
            Linear(dim, dim),
            ReLU(),
            Linear(dim, 1)
        )
        
        # Activations
        self.sigmoid = Sigmoid()

    def forward(self, x, edge_index, batch=None):
        # Layer 1
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        
        # Layer 2
        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        
        # Layer 3
        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        
        # Classifier
        x = self.classifier(x)
        
        # Final Activation to get probability [0, 1]
        x = self.sigmoid(x)
        
        return x

def train(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    
    for data in loader:
        data = data.to(device)
        optimizer.zero_grad()
        
        # Forward pass
        out = model(data.x, data.edge_index, data.batch)
        
        # Prepare Target
        # data.y is now a dense vector [total_num_nodes] with 1s at faulty positions
        # We just need to reshape it to match 'out' [total_num_nodes, 1]
        target = data.y.unsqueeze(1)
        
        # Calculate Loss
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        
    return total_loss / len(loader)

def evaluate(model, loader, device, k=5):
    model.eval()
    correct_top1 = 0
    correct_topk = 0
    total_graphs = 0
    
    with torch.no_grad():
        for data in loader:
            data = data.to(device)
            out = model(data.x, data.edge_index, data.batch)
            
            # We need to process each graph in the batch individually to calculate accuracy
            # data.ptr is useful here
            if hasattr(data, 'ptr'):
                ptr = data.ptr
            else:
                node_counts = torch.bincount(data.batch)
                ptr = torch.cat([torch.tensor([0], device=device), torch.cumsum(node_counts, dim=0)])
            
            for i in range(data.num_graphs):
                # Get range of nodes for this graph
                start = ptr[i]
                end = ptr[i+1]
                
                # Get scores for this graph
                scores = out[start:end].squeeze() # Shape: [num_nodes_in_graph]
                
                # Get true label (local index)
                true_fault_idx = data.y[i].item()
                
                # Top-1 Prediction
                pred_idx = scores.argmax().item()
                if pred_idx == true_fault_idx:
                    correct_top1 += 1
                    
                # Top-k Prediction
                # If graph has fewer than k nodes, take all
                curr_k = min(k, len(scores))
                _, topk_indices = torch.topk(scores, curr_k)
                
                if true_fault_idx in topk_indices.tolist():
                    correct_topk += 1
                
                total_graphs += 1
                
    return correct_top1 / total_graphs, correct_topk / total_graphs

def main():
    # 1. Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 2. Generate Data
    # Create a dummy circuit for demonstration
    dummy_bench = "train_c17.bench"
    content = """
# c17
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
    with open(dummy_bench, "w") as f:
        f.write(content)
        
    print("Generating dataset...")
    G = parse_bench_to_networkx(dummy_bench)
    
    # Generate 500 training samples, 100 testing samples
    full_dataset = generate_dataset(G, num_samples=600)
    train_dataset = full_dataset[:500]
    test_dataset = full_dataset[500:]
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # 3. Initialize Model
    # Infer input dim from first sample
    num_features = train_dataset[0].num_node_features
    model = FaultDiagnosisGNN(num_node_features=num_features).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.BCELoss()
    
    # 4. Training Loop
    epochs = 30
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(epochs):
        loss = train(model, train_loader, optimizer, criterion, device)
        
        if (epoch + 1) % 5 == 0:
            acc1, acc5 = evaluate(model, test_loader, device, k=5)
            print(f"Epoch {epoch+1:02d} | Loss: {loss:.4f} | Test Top-1 Acc: {acc1:.4f} | Test Top-5 Acc: {acc5:.4f}")
            
    # 5. Final Evaluation
    print("\nFinal Evaluation:")
    acc1, acc5 = evaluate(model, test_loader, device, k=5)
    print(f"Top-1 Accuracy: {acc1:.2%}")
    print(f"Top-5 Accuracy: {acc5:.2%}")
    
    # Cleanup
    if os.path.exists(dummy_bench):
        os.remove(dummy_bench)

if __name__ == "__main__":
    main()
