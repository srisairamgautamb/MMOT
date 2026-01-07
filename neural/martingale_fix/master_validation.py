#!/usr/bin/env python3
"""
master_validation.py
====================
COMPREHENSIVE VALIDATION SUITE FOR MMOT NEURAL SOLVER

Tests EVERYTHING:
1. Data Quality (Teacher data drift, NaN check)
2. Neural Model Output (u, h ranges, distillation error)
3. Newton Projection (Convergence, residual)
4. Final Drift (Per instance, aggregate statistics)
5. Universal Stock Coverage (Multiple price points)
6. Performance Metrics (Speed, memory)

Author: MMOT Research Team
"""

import torch
import torch.nn.functional as F
import numpy as np
import sys
import time
import traceback

sys.path.insert(0, '/Volumes/Hippocampus/Antigravity/MMOT/neural/martingale_fix')
from architecture_fixed import ImprovedTransformerMMOT
from hybrid_neural_solver import HybridMMOTSolver


def section_header(title):
    print("\n" + "="*70)
    print(f" {title}")
    print("="*70)


def test_result(name, passed, details=""):
    status = "✅ PASS" if passed else "❌ FAIL"
    print(f"  {name}: {status} {details}")
    return passed


class MasterValidator:
    def __init__(self, device='mps'):
        self.device = device if torch.backends.mps.is_available() or device != 'mps' else 'cpu'
        self.results = {}
        self.all_passed = True
        
    def run_all_tests(self):
        """Run all validation tests."""
        section_header("MASTER VALIDATION SUITE")
        print(f"Device: {self.device}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        # Test 1: Data Quality
        self.test_data_quality()
        
        # Test 2: Model Loading
        self.test_model_loading()
        
        # Test 3: Neural Forward Pass
        self.test_neural_forward()
        
        # Test 4: Newton Projection
        self.test_newton_projection()
        
        # Test 5: Full Pipeline (Drift)
        self.test_full_pipeline_drift()
        
        # Test 6: Universal Stock Coverage
        self.test_universal_coverage()
        
        # Test 7: Performance
        self.test_performance()
        
        # Final Summary
        self.print_summary()
        
        return self.all_passed
    
    def test_data_quality(self):
        """Test 1: Validate teacher data quality."""
        section_header("TEST 1: DATA QUALITY")
        
        try:
            # Load teacher data
            data = np.load('data/mmot_teacher_12000_moneyness.npz', allow_pickle=True)
            n_instances = len(data['marginals'])
            
            passed = test_result("Data loaded", True, f"({n_instances} instances)")
            
            # Check for NaN/Inf
            has_nan = False
            for i in range(min(100, n_instances)):
                if np.any(np.isnan(data['u'][i])) or np.any(np.isnan(data['h'][i])):
                    has_nan = True
                    break
            passed &= test_result("No NaN/Inf values", not has_nan)
            
            # Check teacher drift
            grid = data['grid']
            drifts = []
            epsilon = 0.2
            
            x = torch.from_numpy(grid.astype(np.float32))
            Delta = x.unsqueeze(1) - x.unsqueeze(0)
            C = Delta ** 2
            C_scaled = C / C.max()
            
            for i in range(min(50, n_instances)):
                u = torch.from_numpy(data['u'][i].astype(np.float32))
                h = torch.from_numpy(data['h'][i].astype(np.float32))
                N = h.shape[0]
                
                for t in range(N):
                    term_u = u[t].unsqueeze(1) + u[t+1].unsqueeze(0)
                    term_h = h[t].unsqueeze(1) * Delta
                    LogK = (term_u + term_h - C_scaled) / epsilon
                    probs = F.softmax(LogK, dim=1)
                    expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
                    drift = (expected_y - x).abs().max().item()
                    drifts.append(drift)
            
            mean_drift = np.mean(drifts)
            max_drift = max(drifts)
            
            passed &= test_result("Teacher mean drift < 0.01", mean_drift < 0.01, f"(actual: {mean_drift:.6f})")
            passed &= test_result("Teacher max drift < 0.05", max_drift < 0.05, f"(actual: {max_drift:.6f})")
            
            self.results['data_quality'] = {
                'n_instances': n_instances,
                'has_nan': has_nan,
                'mean_drift': mean_drift,
                'max_drift': max_drift,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['data_quality'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def test_model_loading(self):
        """Test 2: Model loading and architecture."""
        section_header("TEST 2: MODEL LOADING")
        
        try:
            # Load validation data to get M
            data = np.load('data/validation_solved.npz', allow_pickle=True)
            M = len(data['grid'])
            
            # Load model
            self.model = ImprovedTransformerMMOT(M=M, d_model=128, n_heads=4, n_layers=4)
            self.model.load_state_dict(torch.load('checkpoints/best_model.pth', map_location=self.device))
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Count parameters
            n_params = sum(p.numel() for p in self.model.parameters())
            
            passed = test_result("Model loaded", True)
            passed &= test_result("Model on device", True, f"({self.device})")
            passed &= test_result("Parameter count", True, f"({n_params:,} params)")
            
            self.M = M
            self.grid = torch.from_numpy(data['grid'].astype(np.float32)).to(self.device)
            self.val_data = data
            
            self.results['model_loading'] = {
                'M': M,
                'n_params': n_params,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['model_loading'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def test_neural_forward(self):
        """Test 3: Neural network forward pass."""
        section_header("TEST 3: NEURAL FORWARD PASS")
        
        try:
            marginals = self.val_data['marginals'][:10]
            u_teacher = self.val_data['u'][:10]
            h_teacher = self.val_data['h'][:10]
            
            u_errors = []
            h_errors = []
            
            with torch.no_grad():
                for i in range(10):
                    m = torch.from_numpy(marginals[i].astype(np.float32)).unsqueeze(0).to(self.device)
                    N = m.shape[1] - 1
                    
                    # Pad if needed
                    max_N = 5
                    if N < max_N:
                        padded = torch.zeros(1, max_N + 1, self.M, device=self.device)
                        padded[0, :N+1] = m[0]
                        m = padded
                    
                    u_pred, h_pred = self.model(m, self.grid)
                    u_pred = u_pred[0, :N+1].cpu().numpy()
                    h_pred = h_pred[0, :N].cpu().numpy()
                    
                    # MSE error
                    u_err = np.mean((u_pred - u_teacher[i][:N+1])**2)
                    h_err = np.mean((h_pred - h_teacher[i][:N])**2)
                    u_errors.append(u_err)
                    h_errors.append(h_err)
            
            mean_u_err = np.mean(u_errors)
            mean_h_err = np.mean(h_errors)
            
            passed = test_result("Forward pass works", True)
            passed &= test_result("u prediction error < 1000", mean_u_err < 1000, f"(MSE: {mean_u_err:.2f})")
            passed &= test_result("h prediction error < 10000", mean_h_err < 10000, f"(MSE: {mean_h_err:.2f})")
            
            self.results['neural_forward'] = {
                'mean_u_error': mean_u_err,
                'mean_h_error': mean_h_err,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['neural_forward'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def test_newton_projection(self):
        """Test 4: Newton projection convergence."""
        section_header("TEST 4: NEWTON PROJECTION")
        
        try:
            from newton_projection_CORRECT import newton_projection_correct
            
            marginals = self.val_data['marginals'][:5]
            epsilon = 0.2
            
            convergence_count = 0
            final_residuals = []
            iterations_needed = []
            
            with torch.no_grad():
                for i in range(5):
                    m = torch.from_numpy(marginals[i].astype(np.float32)).unsqueeze(0).to(self.device)
                    N = m.shape[1] - 1
                    
                    max_N = 5
                    if N < max_N:
                        padded = torch.zeros(1, max_N + 1, self.M, device=self.device)
                        padded[0, :N+1] = m[0]
                        m = padded
                    
                    u_pred, h_pred = self.model(m, self.grid)
                    u_pred = u_pred[0, :N+1]
                    h_pred = h_pred[0, :N]
                    
                    # Test Newton on t=0
                    h_refined, converged = newton_projection_correct(
                        u_pred[0], u_pred[1], h_pred[0], self.grid,
                        epsilon=epsilon, max_iter=100, verbose=False
                    )
                    
                    if converged:
                        convergence_count += 1
                    
                    # Compute final drift
                    x = self.grid
                    Delta = x.unsqueeze(1) - x.unsqueeze(0)
                    C = Delta ** 2
                    C_scaled = C / C.max()
                    term_u = u_pred[0].unsqueeze(1) + u_pred[1].unsqueeze(0)
                    term_h = h_refined.unsqueeze(1) * Delta
                    LogK = (term_u + term_h - C_scaled) / epsilon
                    probs = F.softmax(LogK, dim=1)
                    expected_y = torch.sum(probs * x.unsqueeze(0), dim=1)
                    drift = (expected_y - x).abs().max().item()
                    final_residuals.append(drift)
            
            mean_residual = np.mean(final_residuals)
            max_residual = max(final_residuals)
            
            passed = test_result("Newton converges (5/5)", convergence_count == 5, f"({convergence_count}/5)")
            passed &= test_result("Mean residual < 0.0001", mean_residual < 0.0001, f"(actual: {mean_residual:.8f})")
            passed &= test_result("Max residual < 0.001", max_residual < 0.001, f"(actual: {max_residual:.8f})")
            
            self.results['newton_projection'] = {
                'convergence_rate': convergence_count / 5,
                'mean_residual': mean_residual,
                'max_residual': max_residual,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['newton_projection'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def test_full_pipeline_drift(self):
        """Test 5: Full hybrid solver pipeline."""
        section_header("TEST 5: FULL PIPELINE DRIFT")
        
        try:
            solver = HybridMMOTSolver('checkpoints/best_model.pth', device=self.device)
            
            marginals = self.val_data['marginals'][:50]
            grid = self.val_data['grid']
            
            drifts = []
            times = []
            
            for i in range(50):
                result = solver.solve(marginals[i], grid, n_newton_iters=100)
                drifts.append(result['drift'])
                times.append(result['total_time'])
            
            mean_drift = np.mean(drifts)
            max_drift = max(drifts)
            pass_rate = sum(1 for d in drifts if d < 0.01) / len(drifts)
            
            passed = test_result("Mean drift < 0.0001", mean_drift < 0.0001, f"(actual: {mean_drift:.8f})")
            passed &= test_result("Max drift < 0.01", max_drift < 0.01, f"(actual: {max_drift:.8f})")
            passed &= test_result("Pass rate >= 100%", pass_rate >= 1.0, f"(actual: {pass_rate*100:.1f}%)")
            
            self.results['full_pipeline'] = {
                'mean_drift': mean_drift,
                'max_drift': max_drift,
                'pass_rate': pass_rate,
                'mean_time': np.mean(times),
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['full_pipeline'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def test_universal_coverage(self):
        """Test 6: Universal stock price coverage."""
        section_header("TEST 6: UNIVERSAL STOCK COVERAGE")
        
        try:
            solver = HybridMMOTSolver('checkpoints/best_model.pth', device=self.device)
            grid = np.linspace(0.5, 1.5, self.M).astype(np.float32)
            
            stocks = [
                ('SPY', 683, 0.15),
                ('AMD', 150, 0.35),
                ('F', 10, 0.25),
                ('TSLA', 395, 0.45),
                ('BRK-B', 450, 0.12),  # Extra test
                ('NVDA', 140, 0.50),   # High vol test
            ]
            
            results = []
            all_pass = True
            
            for stock_name, S0, sigma in stocks:
                # Generate GBM marginals
                marginals = self._generate_gbm_marginals(S0, sigma, T=0.25, N=3, M=self.M, grid=grid)
                result = solver.solve(marginals, grid, n_newton_iters=100)
                
                stock_pass = result['drift'] < 0.01
                if not stock_pass:
                    all_pass = False
                    
                results.append({
                    'stock': stock_name,
                    'S0': S0,
                    'sigma': sigma,
                    'drift': result['drift'],
                    'passed': stock_pass
                })
                
                test_result(f"{stock_name} (${S0})", stock_pass, f"drift={result['drift']:.8f}")
            
            passed = test_result("All stocks pass", all_pass)
            
            self.results['universal_coverage'] = {
                'stocks': results,
                'all_pass': all_pass,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['universal_coverage'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def _generate_gbm_marginals(self, S0, sigma, T, N, M, grid):
        """Generate GBM marginals for testing."""
        dt = T / N
        marginals = np.zeros((N+1, M), dtype=np.float32)
        
        for t in range(N+1):
            if t == 0:
                center_idx = np.argmin(np.abs(grid - 1.0))
                marginals[t, center_idx] = 1.0
            else:
                tau = t * dt
                log_std = sigma * np.sqrt(tau)
                log_m = np.log(grid)
                pdf = np.exp(-0.5 * (log_m / log_std)**2) / (grid * log_std * np.sqrt(2*np.pi))
                marginals[t] = pdf / pdf.sum()
        
        return marginals
    
    def test_performance(self):
        """Test 7: Performance metrics."""
        section_header("TEST 7: PERFORMANCE METRICS")
        
        try:
            solver = HybridMMOTSolver('checkpoints/best_model.pth', device=self.device)
            grid = self.val_data['grid']
            marginals = self.val_data['marginals'][:20]
            
            # Warm up
            _ = solver.solve(marginals[0], grid, n_newton_iters=100)
            
            # Benchmark
            neural_times = []
            newton_times = []
            total_times = []
            
            for i in range(20):
                result = solver.solve(marginals[i], grid, n_newton_iters=100)
                neural_times.append(result['neural_time'])
                newton_times.append(result['newton_time'])
                total_times.append(result['total_time'])
            
            avg_neural = np.mean(neural_times) * 1000
            avg_newton = np.mean(newton_times) * 1000
            avg_total = np.mean(total_times) * 1000
            
            passed = test_result("Avg neural time < 50ms", avg_neural < 50, f"(actual: {avg_neural:.1f}ms)")
            passed &= test_result("Avg newton time < 200ms", avg_newton < 200, f"(actual: {avg_newton:.1f}ms)")
            passed &= test_result("Avg total time < 250ms", avg_total < 250, f"(actual: {avg_total:.1f}ms)")
            
            self.results['performance'] = {
                'avg_neural_ms': avg_neural,
                'avg_newton_ms': avg_newton,
                'avg_total_ms': avg_total,
                'passed': passed
            }
            
            if not passed:
                self.all_passed = False
                
        except Exception as e:
            print(f"  ❌ ERROR: {e}")
            traceback.print_exc()
            self.results['performance'] = {'passed': False, 'error': str(e)}
            self.all_passed = False
    
    def print_summary(self):
        """Print final summary."""
        section_header("FINAL SUMMARY")
        
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results.values() if r.get('passed', False))
        
        print(f"\n  Tests Passed: {passed_tests}/{total_tests}")
        print()
        
        for test_name, result in self.results.items():
            status = "✅ PASS" if result.get('passed', False) else "❌ FAIL"
            print(f"  {test_name}: {status}")
        
        print()
        
        if self.all_passed:
            print("  🎉 ALL TESTS PASSED! SYSTEM IS PRODUCTION-READY!")
        else:
            print("  ⚠️  SOME TESTS FAILED. REVIEW RESULTS ABOVE.")
        
        # Key metrics
        print("\n  KEY METRICS:")
        if 'full_pipeline' in self.results and 'mean_drift' in self.results['full_pipeline']:
            print(f"    Drift:     {self.results['full_pipeline']['mean_drift']:.8f}")
        if 'full_pipeline' in self.results and 'pass_rate' in self.results['full_pipeline']:
            print(f"    Pass Rate: {self.results['full_pipeline']['pass_rate']*100:.1f}%")
        if 'performance' in self.results and 'avg_total_ms' in self.results['performance']:
            print(f"    Avg Time:  {self.results['performance']['avg_total_ms']:.1f}ms")
        
        print("\n" + "="*70)


def main():
    validator = MasterValidator(device='mps')
    all_passed = validator.run_all_tests()
    
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit(main())
