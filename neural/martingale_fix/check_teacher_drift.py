
import numpy as np
import torch
import sys

def main():
    print("Checking TEACHER data (Control)...")
    data = np.load('data/validation_solved.npz', allow_pickle=True)
    
    # Instance 0
    u_teacher = torch.from_numpy(data['u'][0].astype(np.float32)).unsqueeze(0)
    h_teacher = torch.from_numpy(data['h'][0].astype(np.float32)).unsqueeze(0)
    
    grid = torch.from_numpy(data['grid'].astype(np.float32))
    M = len(grid)
    x = grid
    
    # Delta
    # Prior fix: Delta = x_i - x_j  (grid[i] - grid[j])
    x_i = x.unsqueeze(1)
    x_j = x.unsqueeze(0)
    Delta = x_i - x_j
    
    C = Delta ** 2
    C_max = C.max().item()
    C_scaled = C / C_max
    
    epsilon_val = 0.2
    
    print(f"Teacher u range: [{u_teacher.min():.2f}, {u_teacher.max():.2f}]")
    print(f"Teacher h range: [{h_teacher.min():.2f}, {h_teacher.max():.2f}]")
    
    t = 0
    u_t = u_teacher[0, t]
    u_next = u_teacher[0, t+1]
    h_t = h_teacher[0, t] # (M,)
    
    term_u = u_t.unsqueeze(1) + u_next.unsqueeze(0)
    term_h = h_t.unsqueeze(1) * Delta
    
    LogK = (term_u + term_h - C_scaled) / epsilon_val
    
    print(f"Teacher LogK range: [{LogK.min():.2f}, {LogK.max():.2f}]")
    
    probs = torch.softmax(LogK, dim=1)
    ey = torch.sum(probs * x.unsqueeze(0), dim=1)
    drift = (ey - x).abs().max().item()
    
    print(f"Teacher Drift: {drift:.6f}")
    
    if drift > 0.05:
        print("FAIL: Verification script logic is WRONG even for Teacher.")
    else:
        print("SUCCESS: Teacher checks out. Model is just inaccurate.")

if __name__ == "__main__":
    main()
