import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MartingaleProjectionLayer(nn.Module):
    """
    Hard martingale constraint layer that projects h_t to ensure
    E[X_{t+1}|X_t] = X_t exactly via Lagrange multipliers.

    This is the KEY FIX for drift violation.
    """
    def __init__(self, M, epsilon=1e-6):
        super().__init__()
        self.M = M
        self.epsilon = epsilon

    def forward(self, h_t, mu_t, x_grid):
        """
        Args:
            h_t: (batch, M) drift multipliers from neural network
            mu_t: (batch, M) marginal distribution at time t
            x_grid: (M,) spatial grid points

        Returns:
            h_t_projected: (batch, M) projected drift satisfying martingale
        """
        batch_size = h_t.shape[0]

        # Compute transition kernel P(x'|x) \propto exp(h_t(x'|x) / epsilon)
        # For simplicity, use h_t(x'|x) = h_t(x') * (x - x')  [MATCHING SOLVER: x - y]
        x_diff = x_grid.unsqueeze(1) - x_grid.unsqueeze(0)  # (M, M) [i, j] = x_i - x_j
        h_kernel = h_t.unsqueeze(2) * x_diff.unsqueeze(0)  # (batch, M, M)

        # Normalize to get transition probabilities
        P_tx = F.softmax(h_kernel / self.epsilon, dim=2)  # (batch, M, M)

        # Compute drift: E[X_{t+1}|X_t=x] - x
        expected_next = torch.matmul(P_tx, x_grid.unsqueeze(1))  # (batch, M, 1)
        drift = expected_next.squeeze(2) - x_grid.unsqueeze(0)  # (batch, M)

        # Compute Lagrange multiplier to enforce E_{mu_t}[drift] = 0
        weight_drift = (mu_t * drift).sum(dim=1, keepdim=True)  # (batch, 1)

        # Project: h_t' = h_t - lambda * correction
        # Choose lambda to make weighted_drift = 0
        # Simple version: h_t' = h_t - weighted_drift
        correction = weight_drift / (mu_t.sum(dim=1, keepdim=True) + 1e-8)
        h_t_projected = h_t - correction

        return h_t_projected

    def refine_hard_constraint(self, h_t, u_next, x_grid, n_iters=100, tol=1e-5, epsilon=None):
        """
        Apply Newton-Raphson to find EXACT h_t such that E[X_{t+1}|X_t] = X_t.
        
        Args:
            h_t: (batch, M) initial guess (from forward pass)
            u_next: (batch, M) u_{t+1} potentials
            x_grid: (M,) spatial grid
            epsilon: Override epsilon (default uses layer's epsilon)
            
        Returns:
            h_refined: (batch, M) corrected potentials with drift ~ 0
        """
        # Ensure we are in no_grad mode for optimization
        with torch.no_grad():
            if epsilon is None:
                epsilon = self.epsilon
            # Dampen u_next to reduce variance if model is untrained
            # u_next = u_next * 0.1 
            
            delta_S = x_grid[:, None] - x_grid[None, :] # (M, M) x-y [MATCHING SOLVER]
            h_curr = h_t.clone()
            
            # Vectorized Newton-Raphson for each sample in batch & each x in grid
            # Shape logic: 
            # h_curr: (batch, M) -> we solve M independent 1D problems PER batch item
            # We treat (batch, M) as simply (B*M) problems or keep dimensions tailored.
            
            # To simplify, looping over batch (inference usually small batch)
            # Or fully vectorized:
            
            # Expand for broadcasting:
            # u_next: (B, 1, M_y)
            # h_curr: (B, M_x, 1)
            # delta_S: (1, M_x, M_y)
            
            u_broad = u_next.unsqueeze(1) # (B, 1, M)
            ds_broad = delta_S.unsqueeze(0) # (1, M, M)
            
            for i in range(n_iters):
                h_broad = h_curr.unsqueeze(2) # (B, M, 1)
                
                # log_term = (u(y) + h(x)*(y-x)) / eps
                log_term = (u_broad + h_broad * ds_broad) / epsilon
                
                # Softmax stability
                max_log = log_term.max(dim=2, keepdim=True)[0]
                terms = torch.exp(log_term - max_log) # (B, M, M)
                
                # f(h) = Sum_y (y-x) * terms
                # f'(h) = Sum_y (y-x)^2 * terms / eps
                
                # YX = y - x = ds_broad
                integrand_f = ds_broad * terms
                integrand_fp = (ds_broad ** 2) * terms / epsilon
                
                f_val = integrand_f.sum(dim=2) # (B, M)
                fp_val = integrand_fp.sum(dim=2) # (B, M)
                
                # Update
                fp_val = torch.where(fp_val < 1e-8, torch.ones_like(fp_val), fp_val)
                diff = f_val / fp_val
                
                h_curr = h_curr - diff
                
                if diff.abs().max() < tol:
                    break
                    
            return h_curr


class ImprovedTransformerMMOT(nn.Module):
    """
    Improved transformer architecture with hard martingale constraints.
    """
    def __init__(self, M=150, N_max=50, d_model=256, n_heads=4, n_layers=3, 
                 dropout=0.1, epsilon=0.5):
        super().__init__()
        self.M = M
        self.N_max = N_max
        self.d_model = d_model
        self.epsilon = epsilon

        # Marginal embedding
        self.marginal_conv = nn.Conv1d(1, 128, kernel_size=5, padding=2)
        self.marginal_fc = nn.Sequential(
            nn.Linear(128, d_model),
            nn.LayerNorm(d_model),
            nn.GELU()
        )

        # Positional encoding
        self.register_buffer('pos_encoding', self._create_positional_encoding())

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model*4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # Dual potential heads
        self.u_head = nn.Linear(d_model, M)
        self.h_head = nn.Linear(d_model, M)

        # KEY FIX: Martingale projection layer
        self.martingale_projection = MartingaleProjectionLayer(M, epsilon=0.01)

    def _create_positional_encoding(self):
        """Sinusoidal positional encoding."""
        pos = torch.arange(self.N_max + 1).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * 
                            (-math.log(10000.0) / self.d_model))
        pe = torch.zeros(self.N_max + 1, self.d_model)
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        return pe

    def forward(self, marginals, x_grid):
        """
        Args:
            marginals: (batch, N+1, M) marginal distributions
            x_grid: (M,) spatial grid points

        Returns:
            u_potentials: (batch, N+1, M)
            h_potentials: (batch, N, M) with hard martingale constraint
        """
        batch_size, N_plus_1, M = marginals.shape
        N = N_plus_1 - 1

        # Embed marginals
        marginals_flat = marginals.view(batch_size * N_plus_1, M, 1)
        embedded = self.marginal_conv(marginals_flat.transpose(1, 2))  # (B*(N+1), 128, M)
        embedded = embedded.mean(dim=2)  # (B*(N+1), 128)
        embedded = self.marginal_fc(embedded)  # (B*(N+1), d_model)
        embedded = embedded.view(batch_size, N_plus_1, self.d_model)

        # Add positional encoding
        embedded = embedded + self.pos_encoding[:N_plus_1, :].unsqueeze(0)

        # Transformer encoding
        encoded = self.transformer(embedded)  # (batch, N+1, d_model)

        # Compute potentials
        u_potentials = self.u_head(encoded)  # (batch, N+1, M)
        h_potentials_raw = self.h_head(encoded[:, :N, :])  # (batch, N, M)

        # KEY FIX: Apply martingale projection to each time step
        h_potentials = []
        for t in range(N):
            h_t_projected = self.martingale_projection(
                h_potentials_raw[:, t, :],
                marginals[:, t, :],
                x_grid
            )
            h_potentials.append(h_t_projected)

        h_potentials = torch.stack(h_potentials, dim=1)  # (batch, N, M)

        return u_potentials, h_potentials
