import numpy as np
import math
from scipy.stats import norm
import torch

class AugmentedMMOTDataGenerator:
    """
    Generates MMOT instances with diverse marginal structures:
    1. GBM (original)
    2. Merton jump-diffusion (NEW)
    3. Heston stochastic volatility (NEW)
    """

    def __init__(self, M=150, x_min=4000, x_max=10000):
        self.M = M
        self.x_grid = np.linspace(x_min, x_max, M)

    def generate_gbm_marginals(self, S0, sigma, T, N):
        """Original GBM marginals."""
        dt = T / N
        t_grid = np.linspace(0, T, N+1)

        marginals = []
        for t in t_grid:
            if t == 0:
                # Delta distribution at S0
                mu = np.zeros(self.M)
                idx = np.argmin(np.abs(self.x_grid - S0))
                mu[idx] = 1.0
            else:
                # Log-normal density
                mu_log = np.log(S0) - 0.5 * sigma**2 * t
                sigma_log = sigma * np.sqrt(t)
                pdf = norm.pdf(np.log(self.x_grid), mu_log, sigma_log) / self.x_grid
                mu = pdf / (pdf.sum() + 1e-10)

            marginals.append(mu)

        return np.array(marginals)

    def generate_merton_jump_marginals(self, S0, sigma, T, N, 
                                       lambda_jump=5.0, mu_jump=-0.1, sigma_jump=0.15):
        """
        Merton jump-diffusion marginals.

        KEY FIX: This adds multi-modal structure similar to real markets!

        Args:
            lambda_jump: Jump intensity (jumps per year)
            mu_jump: Mean jump size (log-returns)
            sigma_jump: Jump size volatility
        """
        dt = T / N
        t_grid = np.linspace(0, T, N+1)

        marginals = []
        for t in t_grid:
            if t == 0:
                mu = np.zeros(self.M)
                idx = np.argmin(np.abs(self.x_grid - S0))
                mu[idx] = 1.0
            else:
                # Compute PDF as mixture over number of jumps
                max_jumps = int(lambda_jump * t) + 10
                pdf = np.zeros(self.M)

                for k in range(max_jumps):
                    # Probability of k jumps in time t
                    p_k = (lambda_jump * t)**k * np.exp(-lambda_jump * t) / math.factorial(k)

                    # Parameters for log-normal with k jumps
                    mu_log = np.log(S0) - 0.5 * sigma**2 * t + k * mu_jump
                    sigma_log = np.sqrt(sigma**2 * t + k * sigma_jump**2)

                    # Add contribution from k jumps
                    pdf_k = norm.pdf(np.log(self.x_grid), mu_log, sigma_log) / self.x_grid
                    pdf += p_k * pdf_k

                mu = pdf / (pdf.sum() + 1e-10)

            marginals.append(mu)

        return np.array(marginals)

    def generate_heston_marginals(self, S0, v0, T, N, 
                                  kappa=2.0, theta=0.04, sigma_v=0.3, rho=-0.7):
        """
        Heston stochastic volatility marginals (approximation).

        KEY FIX: This adds time-varying spread similar to real markets!

        Args:
            v0: Initial variance
            kappa: Mean-reversion speed
            theta: Long-run variance
            sigma_v: Vol-of-vol
            rho: Correlation between price and volatility
        """
        dt = T / N
        t_grid = np.linspace(0, T, N+1)

        marginals = []
        for t in t_grid:
            if t == 0:
                mu = np.zeros(self.M)
                idx = np.argmin(np.abs(self.x_grid - S0))
                mu[idx] = 1.0
            else:
                # Approximate variance at time t (Feller process mean)
                v_t = theta + (v0 - theta) * np.exp(-kappa * t)

                # Integrated variance
                int_var = theta * t + (v0 - theta) / kappa * (1 - np.exp(-kappa * t))

                # Log-normal approximation with time-varying variance
                mu_log = np.log(S0) - 0.5 * int_var
                sigma_log = np.sqrt(int_var)

                pdf = norm.pdf(np.log(self.x_grid), mu_log, sigma_log) / self.x_grid
                mu = pdf / (pdf.sum() + 1e-10)

            marginals.append(mu)

        return np.array(marginals)

    def generate_augmented_dataset(self, n_instances, save_path='augmented_data.npz'):
        """
        Generate balanced dataset:
        - 50% GBM (keep baseline)
        - 25% Merton jump-diffusion
        - 25% Heston stochastic vol
        """
        all_marginals = []
        all_labels = []  # 0=GBM, 1=Merton, 2=Heston

        n_gbm = n_instances // 2
        n_merton = n_instances // 4
        n_heston = n_instances - n_gbm - n_merton

        print(f"Generating {n_gbm} GBM instances...")
        for i in range(n_gbm):
            # Random parameters
            S0 = np.random.uniform(5000, 8000)
            sigma = np.random.uniform(0.15, 0.35)
            T = np.random.uniform(0.1, 0.5)
            N = 30  # Fixed N for rapid batching

            marginals = self.generate_gbm_marginals(S0, sigma, T, N)
            all_marginals.append(marginals)
            all_labels.append(0)

        print(f"Generating {n_merton} Merton jump-diffusion instances...")
        for i in range(n_merton):
            S0 = np.random.uniform(5000, 8000)
            sigma = np.random.uniform(0.15, 0.30)
            T = np.random.uniform(0.1, 0.5)
            N = 30  # Fixed N for rapid batching
            lambda_jump = np.random.uniform(3, 10)  # 3-10 jumps per year
            mu_jump = np.random.uniform(-0.15, -0.05)  # Negative jumps (crashes)
            sigma_jump = np.random.uniform(0.1, 0.2)

            marginals = self.generate_merton_jump_marginals(
                S0, sigma, T, N, lambda_jump, mu_jump, sigma_jump
            )
            all_marginals.append(marginals)
            all_labels.append(1)

        print(f"Generating {n_heston} Heston stochastic vol instances...")
        for i in range(n_heston):
            S0 = np.random.uniform(5000, 8000)
            v0 = np.random.uniform(0.02, 0.06)  # Initial variance
            T = np.random.uniform(0.1, 0.5)
            N = 30  # Fixed N for rapid batching
            kappa = np.random.uniform(1.0, 3.0)
            theta = np.random.uniform(0.03, 0.05)
            sigma_v = np.random.uniform(0.2, 0.4)

            marginals = self.generate_heston_marginals(
                S0, v0, T, N, kappa, theta, sigma_v
            )
            all_marginals.append(marginals)
            all_labels.append(2)

        # Save dataset
        np.savez(save_path,
                 marginals=all_marginals,
                 labels=np.array(all_labels),
                 x_grid=self.x_grid)

        print(f"✅ Augmented dataset saved to {save_path}")
        print(f"   Total instances: {len(all_marginals)}")
        print(f"   GBM: {n_gbm}, Merton: {n_merton}, Heston: {n_heston}")

        return all_marginals, all_labels


if __name__ == "__main__":
    # RAPID EXECUTION MODE: 2,000 training instances (fits in 1 hour)
    generator = AugmentedMMOTDataGenerator(M=150)
    marginals, labels = generator.generate_augmented_dataset(
        n_instances=2000,
        save_path='mmot_augmented_train.npz'
    )

    # Generate 500 validation instances
    marginals_val, labels_val = generator.generate_augmented_dataset(
        n_instances=500,
        save_path='mmot_augmented_val.npz'
    )
