#!/usr/bin/env python3
"""
regenerate_teacher_data_adaptive.py
===================================
Re-solve ALL 12,000 teacher instances using the Adaptive Epsilon solver.

Target: drift < 0.0001 for publication-quality teacher data.

Estimated time: ~2 hours (0.4s × 12,000 instances + overhead)

Usage:
    python regenerate_teacher_data_adaptive.py

Output:
    data/mmot_teacher_12000_moneyness_HIGHQUALITY.npz
"""

import numpy as np
import sys
import time
import os

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from solve_mmot_adaptive import solve_mmot_adaptive


def regenerate_teacher_data():
    """Re-solve all teacher instances with adaptive epsilon."""
    
    print("\n" + "="*70)
    print("REGENERATING TEACHER DATA WITH ADAPTIVE EPSILON SOLVER")
    print("="*70)
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Load existing teacher data
    input_path = 'data/mmot_teacher_12000_moneyness.npz'
    output_path = 'data/mmot_teacher_12000_moneyness_HIGHQUALITY.npz'
    
    print(f"\nLoading: {input_path}")
    data = np.load(input_path, allow_pickle=True)
    
    marginals_all = data['marginals']
    grid = data['grid']
    models = data['models'] if 'models' in data.files else None
    
    n_instances = len(marginals_all)
    M = len(grid)
    
    print(f"  Instances: {n_instances}")
    print(f"  Grid size: {M}")
    print(f"  Grid range: [{grid.min():.2f}, {grid.max():.2f}]")
    
    # Prepare output arrays
    u_all = []
    h_all = []
    drifts_all = []
    times_all = []
    metadata_all = []
    
    # Solve each instance
    print(f"\nSolving {n_instances} instances...")
    print(f"{'Progress':<15} {'Instance':<10} {'N':<5} {'Drift':<15} {'Time':<10} {'Status'}")
    print("-"*65)
    
    start_total = time.time()
    
    excellent_count = 0
    acceptable_count = 0
    failed_count = 0
    
    for i, marginals in enumerate(marginals_all):
        N = marginals.shape[0] - 1
        
        # Solve with adaptive epsilon
        result = solve_mmot_adaptive(marginals, grid, target_drift=1e-4, verbose=False)
        
        # Store results
        u_all.append(result['u'])
        h_all.append(result['h'])
        drifts_all.append(result['drift'])
        times_all.append(result['total_time'])
        metadata_all.append({
            'drift': result['drift'],
            'total_time': result['total_time'],
            'converged': result['converged'],
            'stage_info': result['stage_info'],
            'model': models[i] if models is not None else None
        })
        
        # Categorize
        if result['drift'] < 0.0001:
            excellent_count += 1
            status = "✅"
        elif result['drift'] < 0.01:
            acceptable_count += 1
            status = "⚠️"
        else:
            failed_count += 1
            status = "❌"
        
        # Progress output
        if (i + 1) % 100 == 0 or i < 10 or i == n_instances - 1:
            elapsed = time.time() - start_total
            eta = (elapsed / (i + 1)) * (n_instances - i - 1)
            pct = (i + 1) / n_instances * 100
            print(f"{pct:5.1f}% ({i+1:5d})  {i+1:<10} {N:<5} {result['drift']:<15.8f} {result['total_time']:<10.1f}s {status}")
            
            if (i + 1) % 1000 == 0:
                print(f"  ETA: {eta/60:.1f} min remaining")
    
    total_time = time.time() - start_total
    
    # Save output
    print(f"\n{'='*65}")
    print(f"SAVING: {output_path}")
    
    np.savez_compressed(
        output_path,
        marginals=np.array(marginals_all),
        grid=grid,
        u=np.array(u_all, dtype=object),
        h=np.array(h_all, dtype=object),
        metadata=np.array(metadata_all, dtype=object),
        models=models if models is not None else np.array([])
    )
    
    # Summary
    print(f"\n{'='*70}")
    print(f"REGENERATION COMPLETE!")
    print(f"{'='*70}")
    print(f"  Output:          {output_path}")
    print(f"  Total instances: {n_instances}")
    print(f"  Total time:      {total_time/60:.1f} minutes")
    print(f"  Avg time/inst:   {total_time/n_instances:.2f}s")
    
    print(f"\n  Quality:")
    print(f"    EXCELLENT (<0.0001): {excellent_count:5d} ({100*excellent_count/n_instances:.1f}%)")
    print(f"    ACCEPTABLE (<0.01):  {acceptable_count:5d} ({100*acceptable_count/n_instances:.1f}%)")
    print(f"    FAILED (>0.01):      {failed_count:5d} ({100*failed_count/n_instances:.1f}%)")
    
    print(f"\n  Drift Statistics:")
    print(f"    Mean drift: {np.mean(drifts_all):.8f}")
    print(f"    Max drift:  {max(drifts_all):.8f}")
    print(f"    Min drift:  {min(drifts_all):.8f}")
    
    if excellent_count == n_instances:
        print(f"\n🎉 PERFECT! 100% of instances achieve drift < 0.0001!")
    elif (excellent_count + acceptable_count) / n_instances >= 0.99:
        print(f"\n✅ EXCELLENT! 99%+ pass rate!")
    else:
        print(f"\n⚠️ Some instances need attention")
    
    print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*70)
    
    return output_path, drifts_all


if __name__ == '__main__':
    regenerate_teacher_data()
