#!/usr/bin/env python3
"""
MMOT Phase 2B: Run All Experiments
==================================

This script runs all experiment tasks and generates all figures/tables.

Usage:
    python run_all_experiments.py          # Run all
    python run_all_experiments.py task7    # Run specific task
    python run_all_experiments.py --quick  # Quick mode (fewer iterations)
"""

import subprocess
import sys
import os
import time

# List of experiment scripts
EXPERIMENTS = {
    'task7': ('experiments/task7_scalability.py', 'Figure 11: Scalability Analysis'),
    'task2': ('experiments/task2_donsker_rate.py', 'Figure 10: Donsker Rate Validation'),
    'task4': ('experiments/task4_heston_comparison.py', 'Table 2: Heston Comparison'),
    'task3': ('experiments/task3_lp_comparison.py', 'Table 3: LP Speedup'),
    'task5': ('experiments/task5_robustness.py', 'Figure 12: Robustness Test'),
    'task1': ('experiments/task1_algorithm_comparison.py', 'Figure 9: Algorithm Comparison'),
    'task6': ('experiments/task6_transaction_costs.py', 'Figure 13: Transaction Costs'),
}

PRIORITY_ORDER = ['task7', 'task2', 'task4', 'task3']  # Must-have first


def run_experiment(task_name):
    """Run a single experiment."""
    if task_name not in EXPERIMENTS:
        print(f"Unknown task: {task_name}")
        return False
    
    script, description = EXPERIMENTS[task_name]
    
    if not os.path.exists(script):
        print(f"⚠️ Script not found: {script}")
        return False
    
    print(f"\n{'='*80}")
    print(f"🔬 RUNNING: {description}")
    print(f"   Script: {script}")
    print('='*80 + "\n")
    
    start = time.time()
    result = subprocess.run([sys.executable, script], capture_output=False)
    elapsed = time.time() - start
    
    if result.returncode == 0:
        print(f"\n✅ {task_name} completed in {elapsed:.1f}s")
        return True
    else:
        print(f"\n❌ {task_name} failed (exit code {result.returncode})")
        return False


def main():
    print("="*80)
    print("🚀 MMOT PHASE 2B: EXPERIMENT RUNNER")
    print("="*80)
    
    # Parse arguments
    args = sys.argv[1:]
    
    if '--quick' in args:
        print("⚡ Quick mode enabled (not implemented yet)")
        args.remove('--quick')
    
    # Determine which experiments to run
    if args:
        tasks = [t for t in args if t in EXPERIMENTS]
    else:
        tasks = PRIORITY_ORDER
    
    if not tasks:
        print("\nUsage:")
        print("  python run_all_experiments.py           # Run priority tasks")
        print("  python run_all_experiments.py task7     # Run specific task")
        print("  python run_all_experiments.py all       # Run all tasks")
        print("\nAvailable tasks:")
        for task, (script, desc) in EXPERIMENTS.items():
            exists = "✓" if os.path.exists(script) else "✗"
            print(f"  {exists} {task}: {desc}")
        return
    
    # Run experiments
    results = {}
    for task in tasks:
        results[task] = run_experiment(task)
    
    # Summary
    print("\n" + "="*80)
    print("📊 SUMMARY")
    print("="*80)
    
    for task, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        _, desc = EXPERIMENTS[task]
        print(f"  {status}: {desc}")
    
    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} experiments passed")
    
    print("\n" + "="*80)
    print("📁 OUTPUT FILES")
    print("="*80)
    
    output_dir = 'figures/phase2b'
    if os.path.exists(output_dir):
        files = os.listdir(output_dir)
        for f in sorted(files):
            size = os.path.getsize(os.path.join(output_dir, f))
            print(f"  {f} ({size/1024:.1f} KB)")
    else:
        print("  (no output files yet)")


if __name__ == "__main__":
    main()
