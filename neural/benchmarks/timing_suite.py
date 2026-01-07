import time
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
import sys
import yaml
import json
import argparse

# Add project root
sys.path.append('/Volumes/Hippocampus/Antigravity/MMOT')

from neural.models.architecture import create_model

def load_validation_data(data_dir, device):
    files = sorted([f for f in Path(data_dir).glob('*.npz') if not f.name.startswith('._')])
    instances = []
    for f in files:
        data = np.load(f, allow_pickle=True)
        marginals = torch.from_numpy(data['marginals']).float().to(device)
        instances.append(marginals)
    return instances

def benchmark_solver(model, instances, n_repeats=10, device='cpu'):
    """Run systematic timing benchmarks"""
    times = []
    model.eval()
    
    # Warmup
    print("  Warming up...")
    with torch.no_grad():
        for _ in range(5):
            _ = model(instances[0].unsqueeze(0))
    
    print(f"  Benchmarking {len(instances)} instances x {n_repeats} repeats...")
    
    with torch.no_grad():
        for i, instance in enumerate(instances):
            # Batch dim
            inp = instance.unsqueeze(0)
            
            for _ in range(n_repeats):
                if device.type == 'mps':
                    torch.mps.synchronize()
                elif device.type == 'cuda':
                    torch.cuda.synchronize()
                
                t0 = time.perf_counter()
                
                _ = model(inp)
                
                if device.type == 'mps':
                    torch.mps.synchronize()
                elif device.type == 'cuda':
                    torch.cuda.synchronize()
                    
                t1 = time.perf_counter()
                times.append((t1 - t0) * 1000)  # ms
                
    times = np.array(times)
    
    stats = {
        'mean': float(times.mean()),
        'std': float(times.std()),
        'median': float(np.median(times)),
        'p95': float(np.percentile(times, 95)),
        'p05': float(np.percentile(times, 5)),
        'min': float(times.min()),
        'max': float(times.max()),
        'samples': len(times)
    }
    return stats

def run_benchmarks(args):
    print("="*60)
    print("TIMING BENCHMARK SUITE")
    print("="*60)
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load Model
    with open('neural/configs/production_training.yaml', 'r') as f:
        config = yaml.safe_load(f)
    with open('neural/configs/default.yaml', 'r') as f:
        defaults = yaml.safe_load(f)
    
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, dict):
                d[k] = deep_update(d.get(k, {}), v)
            else:
                d[k] = v
        return d
    final_config = deep_update(defaults, config)
    
    model = create_model(final_config['model'])
    model.to(device)
    
    ckpt_path = 'neural/checkpoints/best_model.pt'
    if Path(ckpt_path).exists():
        print(f"Loading weights: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        print("⚠️ No checkpoint found. Benchmarking initialized weights.")
        
    # Load Data
    val_dir = 'neural/data/val'
    instances = load_validation_data(val_dir, device)
    if not instances:
        print("Val empty, using Train")
        instances = load_validation_data('neural/data/train', device)
    
    # Limit instances
    instances = instances[:args.n_instances]
    
    # Run
    stats = benchmark_solver(model, instances, n_repeats=args.n_repeats, device=device)
    
    print("-" * 60)
    print("RESULTS (ms per instance)")
    print("-" * 60)
    print(f"Mean ± Std: {stats['mean']:.2f} ± {stats['std']:.2f} ms")
    print(f"Median:     {stats['median']:.2f} ms")
    print(f"95th %ile:  {stats['p95']:.2f} ms")
    print("-" * 60)
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Saved to {args.output}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_instances', type=int, default=20)
    parser.add_argument('--n_repeats', type=int, default=10)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    
    run_benchmarks(args)
