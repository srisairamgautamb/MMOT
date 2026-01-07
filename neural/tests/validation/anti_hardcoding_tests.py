#!/usr/bin/env python3
"""
Anti-Hardcoding Tests: Lipschitz Continuity & Interpolation
============================================================
Proves: Model responds smoothly to input changes (not a lookup table)
Method: 
  - Perturb marginals by ε, measure output change
  - Test interpolated inputs between training examples

This is Tests 4 & 5 from the validation plan.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from tqdm import tqdm

from neural.models.architecture import NeuralDualSolver
from neural.inference.pricer import NeuralPricer


def test_lipschitz_continuity(model, data_dir: str, epsilon: float = 0.01, 
                               n_trials: int = 10, device: str = 'mps'):
    """
    Test 4: Lipschitz Continuity
    
    If model is a lookup table, small input change → discrete jump
    If model is learned function, small input change → small output change
    
    We measure the Lipschitz constant: |f(x+ε) - f(x)| / |ε|
    A genuine neural network should have L < 10
    A lookup table would have L > 50 (discrete jumps)
    """
    print("\n" + "="*60)
    print("TEST 4: LIPSCHITZ CONTINUITY")
    print("="*60)
    
    # Load test file
    data_files = sorted(Path(data_dir).glob('*.npz'))[:20]  # Test on 20 instances
    
    grid = torch.linspace(50, 200, 150).to(device)
    pricer = NeuralPricer(model, grid, epsilon=1.0, device=device)
    
    all_sensitivities = []
    
    for file in tqdm(data_files, desc="Testing Lipschitz"):
        data = np.load(file, allow_pickle=True)
        marginals = torch.from_numpy(data['marginals']).float().to(device)
        
        # Original prediction
        with torch.no_grad():
            u_0, h_0 = model(marginals.unsqueeze(0))
        
        # Compute original price
        paths_0 = pricer.sample_paths_from_potentials(
            u_0[0], h_0[0], marginals, num_paths=1000
        )
        price_0 = paths_0.mean(dim=1).mean().item()
        
        sensitivities = []
        
        for trial in range(n_trials):
            # Add small random noise
            noise = torch.randn_like(marginals) * epsilon
            marginals_perturbed = marginals + noise
            
            # Renormalize (marginals must sum to 1)
            marginals_perturbed = F.softmax(marginals_perturbed, dim=-1)
            
            # Perturbed prediction
            with torch.no_grad():
                u_1, h_1 = model(marginals_perturbed.unsqueeze(0))
            
            # Compute perturbed price
            paths_1 = pricer.sample_paths_from_potentials(
                u_1[0], h_1[0], marginals_perturbed, num_paths=1000
            )
            price_1 = paths_1.mean(dim=1).mean().item()
            
            # Measure sensitivity
            input_change = torch.norm(noise).item()
            output_change = abs(price_1 - price_0)
            
            if input_change > 1e-8:
                sensitivity = output_change / input_change
            else:
                sensitivity = 0
            
            sensitivities.append(sensitivity)
        
        all_sensitivities.extend(sensitivities)
    
    mean_sensitivity = np.mean(all_sensitivities)
    std_sensitivity = np.std(all_sensitivities)
    
    print(f"\nLipschitz constant estimate: {mean_sensitivity:.2f} ± {std_sensitivity:.2f}")
    print(f"Min: {np.min(all_sensitivities):.2f}, Max: {np.max(all_sensitivities):.2f}")
    
    # PASS CRITERION
    if mean_sensitivity < 10:
        print("✅ PASS: Smooth response (L < 10, genuine learning)")
        passed = True
    elif mean_sensitivity < 30:
        print("⚠️ MARGINAL: Moderately smooth (10 < L < 30)")
        passed = True
    else:
        print("❌ FAIL: Jumpy response (L > 30, possible hardcoding)")
        passed = False
    
    return {
        'test_name': 'Lipschitz Continuity',
        'mean_lipschitz': float(mean_sensitivity),
        'std_lipschitz': float(std_sensitivity),
        'min_lipschitz': float(np.min(all_sensitivities)),
        'max_lipschitz': float(np.max(all_sensitivities)),
        'epsilon': epsilon,
        'n_trials': n_trials,
        'passed': passed
    }


def test_interpolation(model, data_dir: str, device: str = 'mps'):
    """
    Test 5: Interpolation Test
    
    Take pairs of training instances, create average marginals
    Test if model handles interpolated inputs reasonably
    
    A genuine model should interpolate smoothly
    An overfit model would fail on interpolated inputs
    """
    print("\n" + "="*60)
    print("TEST 5: INTERPOLATION")
    print("="*60)
    
    # Load pairs of training files
    data_files = sorted(Path(data_dir).glob('*.npz'))[:40]
    
    grid = torch.linspace(50, 200, 150).to(device)
    pricer = NeuralPricer(model, grid, epsilon=1.0, device=device)
    
    interpolation_results = []
    
    for i in tqdm(range(0, len(data_files) - 1, 2), desc="Testing Interpolation"):
        data1 = np.load(data_files[i], allow_pickle=True)
        data2 = np.load(data_files[i + 1], allow_pickle=True)
        
        marg1 = torch.from_numpy(data1['marginals']).float().to(device)
        marg2 = torch.from_numpy(data2['marginals']).float().to(device)
        
        # Only interpolate if same shape
        if marg1.shape != marg2.shape:
            continue
        
        # Average their marginals (interpolation)
        marginals_interp = 0.5 * (marg1 + marg2)
        marginals_interp = F.softmax(marginals_interp, dim=-1)  # Renormalize
        
        # Get predictions for originals and interpolated
        with torch.no_grad():
            u1, h1 = model(marg1.unsqueeze(0))
            u2, h2 = model(marg2.unsqueeze(0))
            u_interp, h_interp = model(marginals_interp.unsqueeze(0))
        
        # Compute prices for all three
        paths1 = pricer.sample_paths_from_potentials(u1[0], h1[0], marg1, num_paths=1000)
        paths2 = pricer.sample_paths_from_potentials(u2[0], h2[0], marg2, num_paths=1000)
        paths_interp = pricer.sample_paths_from_potentials(
            u_interp[0], h_interp[0], marginals_interp, num_paths=1000
        )
        
        price1 = paths1.mean(dim=1).mean().item()
        price2 = paths2.mean(dim=1).mean().item()
        price_interp = paths_interp.mean(dim=1).mean().item()
        
        # Expected: interpolated price should be roughly between the two
        expected_price = 0.5 * (price1 + price2)
        
        if expected_price > 0.01:
            deviation = abs(price_interp - expected_price) / expected_price * 100
        else:
            deviation = 0
        
        interpolation_results.append({
            'pair': (data_files[i].name, data_files[i+1].name),
            'price1': price1,
            'price2': price2,
            'price_interp': price_interp,
            'expected_price': expected_price,
            'deviation_pct': deviation
        })
    
    mean_deviation = np.mean([r['deviation_pct'] for r in interpolation_results])
    std_deviation = np.std([r['deviation_pct'] for r in interpolation_results])
    
    print(f"\nInterpolation deviation: {mean_deviation:.2f}% ± {std_deviation:.2f}%")
    
    # PASS CRITERION
    if mean_deviation < 20:
        print("✅ PASS: Handles interpolation well (deviation < 20%)")
        passed = True
    else:
        print("❌ FAIL: Poor interpolation (overfit suspected)")
        passed = False
    
    return {
        'test_name': 'Interpolation',
        'mean_deviation': float(mean_deviation),
        'std_deviation': float(std_deviation),
        'n_pairs': len(interpolation_results),
        'passed': passed,
        'details': interpolation_results[:10]  # First 10 for brevity
    }


def run_all_anti_hardcoding_tests(model_path: str, data_dir: str, 
                                   device: str = 'mps', results_path: str = None):
    """Run all anti-hardcoding tests."""
    
    # Load model
    print(f"Loading model from: {model_path}")
    model = NeuralDualSolver(grid_size=150)  # Match checkpoint
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    # Run tests
    results = {}
    
    # Test 4: Lipschitz
    results['lipschitz'] = test_lipschitz_continuity(
        model, data_dir, epsilon=0.01, n_trials=10, device=device
    )
    
    # Test 5: Interpolation
    results['interpolation'] = test_interpolation(model, data_dir, device=device)
    
    # Summary
    print("\n" + "="*60)
    print("ANTI-HARDCODING TESTS SUMMARY")
    print("="*60)
    all_passed = all([results['lipschitz']['passed'], results['interpolation']['passed']])
    
    print(f"Lipschitz:     {'✅ PASS' if results['lipschitz']['passed'] else '❌ FAIL'}")
    print(f"Interpolation: {'✅ PASS' if results['interpolation']['passed'] else '❌ FAIL'}")
    print(f"\nOverall: {'✅ MODEL IS GENUINE' if all_passed else '❌ POSSIBLE HARDCODING'}")
    
    results['overall_passed'] = all_passed
    
    # Save results
    if results_path:
        Path(results_path).parent.mkdir(parents=True, exist_ok=True)
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {results_path}")
    
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Anti-Hardcoding Tests')
    parser.add_argument('--model', type=str, default='neural/checkpoints/best_model.pt')
    parser.add_argument('--data_dir', type=str, default='neural/data/val')
    parser.add_argument('--device', type=str, default='mps')
    parser.add_argument('--results', type=str, default='neural/results/validation/anti_hardcoding.json')
    
    args = parser.parse_args()
    
    run_all_anti_hardcoding_tests(
        args.model, 
        args.data_dir, 
        args.device, 
        args.results
    )
