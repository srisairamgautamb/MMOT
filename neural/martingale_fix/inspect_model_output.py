
import torch
import numpy as np
import sys
import pandas as pd

# Import modules
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT
from train_with_teacher_moneyness import MoneynessDataset

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Inspecting on {device}...")
    
    # Load Data
    data = np.load('data/validation_solved.npz', allow_pickle=True)
    marginals = data['marginals']
    grid = data['grid']
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)
    M = len(grid)
    
    # Load Model
    model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()
    
    # Inspect Instance 0
    i = 0
    m = torch.from_numpy(marginals[i].astype(np.float32)).unsqueeze(0).to(device)
    
    with torch.no_grad():
        u_pred, h_pred = model(m, grid_torch)
        
    u = u_pred[0].cpu().numpy()
    h = h_pred[0].cpu().numpy()
    
    print("\n--- RAW POTENTIALS ---")
    print(f"u (N+1 x M): min={u.min():.4f}, max={u.max():.4f}, mean={u.mean():.4f}")
    print(f"h (N x M):   min={h.min():.4f}, max={h.max():.4f}, mean={h.mean():.4f}")
    
    # Calculate Transition P for t=0
    t = 0
    epsilon = 0.2
    
    # Grid diffs (Delta = x_i - x_j) matching verifying script
    x = grid_torch
    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    Delta = x_i - x_j
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    u_t = u_pred[0, t]
    u_next = u_pred[0, t+1]
    h_t = h_pred[0, t]
    
    term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
    term_h = h_t.unsqueeze(1) * Delta
    
    LogK = (term_u + term_h - C_scaled) / epsilon
    probs = torch.softmax(LogK, dim=1) # dim 1 is y (columns)
    
    # Calculate Drift Vector
    ey = torch.sum(probs * x.unsqueeze(0), dim=1)
    drift_vector = (ey - x).cpu().numpy()
    abs_drift = np.abs(drift_vector)
    
    print("\n--- DRIFT ANALYSIS (t=0) ---")
    print(f"LogK Range: [{LogK.min().item():.2f}, {LogK.max().item():.2f}]")
    print(f"P(y|x) Row Sums: {probs.sum(dim=1).mean().item():.4f}")
    
    print("\nDrift at specific x points:")
    df = pd.DataFrame({
        'x': grid[::15], # Sample every 15th point
        'Drift': drift_vector[::15],
        'AbsDrift': abs_drift[::15]
    })
    print(df.to_string(index=False))
    
    max_drift = abs_drift.max()
    print(f"\nMAX DRIFT = {max_drift:.6f}")
    
    # Diagnosis
    if max_drift > 0.9:
        print("\nDIAGNOSIS: The model is pushing mass to the edges due to high potentials.")
        print("Likely cause: Neural Network learned to minimize Loss by exploiting the unconstrained nature of u/h.")
    
if __name__ == "__main__":
    main()
