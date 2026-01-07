
import torch
import numpy as np
import sys
import os

# Import modules
sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT, MartingaleProjectionLayer
from train_with_teacher_moneyness import MoneynessDataset

def main():
    device = 'mps' if torch.backends.mps.is_available() else 'cpu'
    print(f"Verifying on {device}...")
    
    # Load Data
    data_path = 'data/validation_solved.npz'
    data = np.load(data_path, allow_pickle=True)
    marginals = data['marginals']
    grid = data['grid']
    grid_torch = torch.from_numpy(grid.astype(np.float32)).to(device)
    
    M = len(grid)
    
    # Load Model
    model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4).to(device)
    model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=device))
    model.eval()
    
    # Helper: Log Kernel
    x = grid_torch
    x_i = x.unsqueeze(1) # (M, 1)
    x_j = x.unsqueeze(0) # (1, M)
    Delta = x_i - x_j # Corrected Definition (x-y) matchiing solver
    
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    epsilon_val = 0.2
    
    drifts_raw = []
    drifts_proj = []
    
    print("Checking first 100 validation instances...")
    
    proj_layer = MartingaleProjectionLayer(M=M, epsilon=epsilon_val).to(device)
    
    with torch.no_grad():
        for i in range(100):
            m = torch.from_numpy(marginals[i].astype(np.float32)).unsqueeze(0).to(device) # (1, N+1, M)
            N = m.shape[1] - 1
            
            # Forward
            u_pred, h_pred = model(m, grid_torch)
            
            if i == 0:
                print(f"DEBUG: u range [{u_pred.min():.2f}, {u_pred.max():.2f}]")
                print(f"DEBUG: h range [{h_pred.min():.2f}, {h_pred.max():.2f}]")

            # --- Check Raw Drift ---
            for t in range(N):
                u_t = u_pred[0, t]
                u_next = u_pred[0, t+1]
                h_t = h_pred[0, t]
                
                term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
                term_h = h_t.unsqueeze(1) * Delta
                
                LogK = (term_u + term_h - C_scaled) / epsilon_val
                
                logits = LogK
                probs = torch.softmax(logits, dim=1)
                
                if i == 0 and t == 0:
                     print(f"DEBUG [0,0]: LogK range [{LogK.min():.2f}, {LogK.max():.2f}]")
                     row0_probs = probs[0,:]
                     print(f"DEBUG [0,0]: Row0 Max Prob: {row0_probs.max():.4f} at idx {row0_probs.argmax()}")
                     ey = torch.sum(probs * x.unsqueeze(0), dim=1)
                     print(f"DEBUG [0,0]: Expected Y range [{ey.min():.4f}, {ey.max():.4f}]")
                
                expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
                drift = (expected_y - x).abs().max().item()
                drifts_raw.append(drift)
                
                # --- Check Projected Drift ---
                # Apply Newton refinement with MORE ITERATIONS for sharp epsilon
                h_initial = h_t.unsqueeze(0)
                h_refined = proj_layer.refine_hard_constraint(h_initial, m[:,t,:], grid_torch, epsilon=epsilon_val, n_iters=500)
                h_refined = h_refined[0]
                
                # DEBUG PROJECTION
                if i == 0 and t == 0:
                    diff_h = (h_refined - h_t).abs().max().item()
                    print(f"DEBUG [0,0]: Newton Correction |h_new - h_old| max = {diff_h:.4f}")
                    
                    term_h_new = h_refined.unsqueeze(1) * Delta
                    logits_new = (u_next.unsqueeze(0) + term_h_new - C_scaled) / epsilon_val
                    probs_new = torch.softmax(logits_new, dim=1)
                    ey_new = torch.sum(probs_new * x.unsqueeze(0), dim=1)
                    print(f"DEBUG [0,0]: Projected Expected Y range [{ey_new.min():.4f}, {ey_new.max():.4f}]")
                
                # Recalculate drift with h_refined
                term_h_new = h_refined.unsqueeze(1) * Delta
                logits_new = (u_next.unsqueeze(0) + term_h_new - C_scaled) / epsilon_val
                probs_new = torch.softmax(logits_new, dim=1)
                expected_y_new = torch.sum(probs_new * x.unsqueeze(0), dim=1)
                drift_proj = (expected_y_new - x).abs().max().item()
                drifts_proj.append(drift_proj)

    mean_raw = np.mean(drifts_raw)
    mean_proj = np.mean(drifts_proj)
    
    print("\nRESULTS:")
    print(f"Mean Raw Drift:       {mean_raw:.6f}")
    print(f"Mean Projected Drift: {mean_proj:.6f}")
    
    if mean_proj < 0.05:
        print("SUCCESS: Projected drift is acceptable! ✅")
    else:
        print("WARNING: High drift even after projection.")

if __name__ == "__main__":
    main()
