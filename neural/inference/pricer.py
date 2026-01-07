import torch
import torch.nn.functional as F
import numpy as np
import time

class NeuralPricer:
    def __init__(self, model, grid, epsilon=1.0, device='cpu'):
        self.model = model
        self.grid = grid
        self.epsilon = epsilon
        self.device = device
        self.model.eval()
        
    def get_gibbs_kernel(self, u_next, h_curr, x_curr_idx):
        """
        Compute P(x_{t+1} | x_t) using the Gibbs form.
        
        Args:
            u_next: Potential at t+1 [Grid]
            h_curr: Potential at t [Grid] (scalar value h(x_t))
            x_curr_idx: Index of current state x_t
        
        Returns:
            probs: Probability distribution over x_{t+1} [Grid]
        """
        S_t = self.grid[x_curr_idx]
        delta_S = self.grid - S_t  # [Grid]
        
        # Log kernel: (u(y) + h(x)(y-x)) / epsilon
        # h_curr is the value of h at x_t.
        # u_next is the vector of u values at all y.
        
        log_kernel = (u_next + h_curr * delta_S) / self.epsilon
        probs = F.softmax(log_kernel, dim=0)
        return probs

    def sample_paths(self, marginals, num_paths=1000):
        """
        Sample paths using the learned kernels.
        
        Args:
            marginals: [N+1, M] Input marginals
            num_paths: Number of Monte Carlo paths
            
        Returns:
            paths: [num_paths, N+1] Price paths
        """
        with torch.no_grad():
            # 1. Get Potentials
            # Add batch dim [1, N+1, M]
            u_pred, h_pred = self.model(marginals.unsqueeze(0).to(self.device))
            
            # Remove batch dim
            u_pred = u_pred[0]  # [N+1, M]
            h_pred = h_pred[0]  # [N, M]
            
        N = h_pred.shape[0]
        paths = torch.zeros(num_paths, N+1, device=self.device)
        
        # Initialize X_0 based on marginal 0
        # For simplicity, sample from the categorical distribution of mu_0
        mu_0 = marginals[0]
        x_indices = torch.multinomial(mu_0, num_paths, replacement=True)
        paths[:, 0] = self.grid[x_indices]
        
        # Step through time
        for t in range(N):
            # We need to sample next step for EACH path.
            # Vectorized kernel computation?
            # P(y|x) depends on x through h_t(x).
            
            # Gather h_t(x_path) for all paths
            # x_indices are the grid indices of current X_t
            # We need to convert path values back to indices if they drifted?
            # No, we simulate ON THE GRID.
            
            current_x_indices = x_indices
            
            # Get h_t values for all current positions
            # h_pred[t] is [M] vector.
            h_vals = h_pred[t][current_x_indices]  # [num_paths]
            
            # Get u_{t+1} vector
            u_next = u_pred[t+1]  # [M]
            
            # Compute Kernels efficiently
            # We have [num_paths] different kernels because h_vals differ.
            # LogKernel[p, y] = (u_next[y] + h_vals[p] * (S[y] - S[x_p])) / eps
            
            # Expand for broadcasting
            # S[y] [1, M]
            # S[x_p] [P, 1]
            S_y = self.grid.unsqueeze(0)        # [1, M]
            S_x = self.grid[current_x_indices].unsqueeze(1) # [P, 1]
            H_x = h_vals.unsqueeze(1)           # [P, 1]
            U_y = u_next.unsqueeze(0)           # [1, M]
            
            # BUG FIX: Add u_t (current potential) and cost term
            u_curr = u_pred[t][current_x_indices]  # [P]
            U_x = u_curr.unsqueeze(1)           # [P, 1]
            cost = (S_y - S_x) ** 2             # [P, M] - quadratic transport cost
            
            # CORRECTED Gibbs kernel: includes u_t, u_{t+1}, h_t * delta_S, and cost
            log_K = (U_x + U_y + H_x * (S_y - S_x) - cost) / self.epsilon
            probs = F.softmax(log_K, dim=1)     # [P, M]
            
            # Sample next indices
            new_indices = torch.multinomial(probs, 1).squeeze()
            paths[:, t+1] = self.grid[new_indices]
            x_indices = new_indices
            
        return paths

    def sample_paths_from_potentials(self, u_pred, h_pred, marginals, num_paths=1000):
        """Sample paths given explicit potentials (e.g. from classical solver)"""
        # u_pred: [N+1, M]
        # h_pred: [N, M]
        N = h_pred.shape[0]
        paths = torch.zeros(num_paths, N+1, device=self.device)
        
        # Initialize X_0
        mu_0 = marginals[0]
        x_indices = torch.multinomial(mu_0, num_paths, replacement=True)
        paths[:, 0] = self.grid[x_indices]
        
        for t in range(N):
            current_x_indices = x_indices
            h_vals = h_pred[t][current_x_indices]
            u_next = u_pred[t+1]
            
            S_y = self.grid.unsqueeze(0)        # [1, M]
            S_x = self.grid[current_x_indices].unsqueeze(1) # [P, 1]
            H_x = h_vals.unsqueeze(1)           # [P, 1]
            U_y = u_next.unsqueeze(0)           # [1, M]
            
            # BUG FIX: Add u_t (current potential) and cost term
            u_curr = u_pred[t][current_x_indices]  # [P]
            U_x = u_curr.unsqueeze(1)           # [P, 1]
            cost = (S_y - S_x) ** 2             # [P, M] - quadratic transport cost
            
            # CORRECTED Gibbs kernel: includes u_t, u_{t+1}, h_t * delta_S, and cost
            log_K = (U_x + U_y + H_x * (S_y - S_x) - cost) / self.epsilon
            probs = F.softmax(log_K, dim=1)     # [P, M]
            
            new_indices = torch.multinomial(probs, 1).squeeze()
            paths[:, t+1] = self.grid[new_indices]
            x_indices = new_indices
            
        return paths

    def price_asian_call(self, marginals, strike, num_paths=10000, T=1.0, r=0.0):
        paths = self.sample_paths(marginals, num_paths)
        average_price = paths.mean(dim=1)
        payoff = F.relu(average_price - strike)
        discount = np.exp(-r * T)
        return discount * payoff.mean().item()
        
    def price_with_potentials(self, u, h, marginals, strike, num_paths=10000, T=1.0, r=0.0):
        paths = self.sample_paths_from_potentials(u, h, marginals, num_paths)
        average_price = paths.mean(dim=1)
        payoff = F.relu(average_price - strike)
        discount = np.exp(-r * T)
        return discount * payoff.mean().item()

    def compute_dual_val(self, marginals):
        """
        Compute the dual objective value: sum <u_t, mu_t>
        """
        with torch.no_grad():
            u_pred, _ = self.model(marginals.unsqueeze(0).to(self.device))
            u_pred = u_pred[0] # [N+1, M]
        
        # Dual value = sum_t dot(u_t, mu_t)
        # u is normalized? No, model output is denormalized.
        # marginals are probability vectors.
        
        val = 0.0
        for t in range(u_pred.shape[0]):
            val += torch.dot(u_pred[t], marginals[t].to(self.device)).item()
            
        return val
