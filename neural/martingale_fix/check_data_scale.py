
import numpy as np

data = np.load('data/validation_solved.npz', allow_pickle=True)
u = data['u']
h = data['h']

# Flatten
u_flat = np.concatenate([x.flatten() for x in u])
h_flat = np.concatenate([x.flatten() for x in h])

print(f"u stats: Mean={u_flat.mean():.2f}, Std={u_flat.std():.2f}, Min={u_flat.min():.2f}, Max={u_flat.max():.2f}")
print(f"h stats: Mean={h_flat.mean():.2f}, Std={h_flat.std():.2f}, Min={h_flat.min():.2f}, Max={h_flat.max():.2f}")

# Calculate MSE if prediction was zero
mse_u_zero = np.mean(u_flat**2)
mse_h_zero = np.mean(h_flat**2)
print(f"Baseline MSE (Zero Pred): u={mse_u_zero:.2f}, h={mse_h_zero:.2f}, Total={mse_u_zero+mse_h_zero:.2f}")
