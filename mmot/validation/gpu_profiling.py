"""
GPU PROFILING AND JAX CONFIGURATION VERIFICATION
=================================================
This script checks JAX backend configuration and measures performance.
"""
import jax
import jax.numpy as jnp
import time
import platform
import sys

def check_jax_configuration():
    """Check JAX backend and device configuration."""
    print("="*70)
    print("JAX CONFIGURATION CHECK")
    print("="*70)
    
    # System info
    print(f"\n📍 SYSTEM INFO:")
    print(f"  Python: {sys.version.split()[0]}")
    print(f"  Platform: {platform.system()} {platform.machine()}")
    
    # JAX version
    print(f"\n📦 JAX VERSION:")
    print(f"  jax: {jax.__version__}")
    try:
        import jaxlib
        print(f"  jaxlib: {jaxlib.__version__}")
    except ImportError:
        print("  jaxlib: (import failed)")
    
    # Backend and devices
    print(f"\n🔧 BACKEND CONFIGURATION:")
    backend = jax.default_backend()
    print(f"  Default backend: {backend}")
    
    print(f"\n🖥️  DEVICES:")
    devices = jax.devices()
    for i, dev in enumerate(devices):
        print(f"  [{i}] {dev}")
    
    # Check for GPU
    print(f"\n🎮 GPU STATUS:")
    try:
        gpu_devices = jax.devices('gpu')
        print(f"  GPU devices found: {len(gpu_devices)}")
        for dev in gpu_devices:
            print(f"    - {dev}")
    except RuntimeError:
        print("  ⚠️  No GPU devices available")
        print("  Running on CPU (still functional, but not GPU-accelerated)")
    
    # Check for Metal (Apple Silicon)
    print(f"\n🍎 APPLE METAL STATUS:")
    if backend == 'metal' or 'metal' in str(jax.devices()).lower():
        print("  ✅ Metal backend detected (Apple Silicon GPU)")
    elif platform.machine() == 'arm64' and platform.system() == 'Darwin':
        print("  ⚠️  Apple Silicon detected but Metal not active")
        print("  To enable: pip install jax-metal")
    else:
        print("  N/A (not Apple Silicon)")
    
    return backend

def run_performance_benchmark():
    """Run performance benchmarks on MMOT solver."""
    from mmot.core import solver, ops
    
    print("\n" + "="*70)
    print("PERFORMANCE BENCHMARK")
    print("="*70)
    
    backend = jax.default_backend()
    
    benchmarks = []
    
    # Test different problem sizes
    test_sizes = [
        (5, 50, "Small"),
        (10, 100, "Medium"),
        (30, 100, "Large"),
        (50, 100, "XL (Spec)"),
    ]
    
    for N, M, label in test_sizes:
        print(f"\n📊 {label}: N={N}, M={M}")
        
        # Setup
        x_grid = jnp.linspace(-3.0, 3.0, M)
        Delta = x_grid[None, :] - x_grid[:, None]
        C = 0.5 * (Delta ** 2)
        
        marginals = []
        for t in range(N + 1):
            t_scaled = 2.0 * t / N
            sigma = jnp.sqrt(0.2 + 0.15 * t_scaled)
            pdf = jnp.exp(-0.5 * (x_grid / sigma)**2)
            pdf = pdf / jnp.sum(pdf)
            marginals.append(pdf)
        marginals = jnp.array(marginals)
        
        # Warmup (JIT compilation)
        _ = solver.solve_mmot(marginals, C, x_grid, max_iter=10, epsilon=0.05, damping=0.8)
        jax.block_until_ready(_)
        
        # Benchmark
        start = time.time()
        u, h, iters = solver.solve_mmot(
            marginals, C, x_grid,
            max_iter=2000,
            epsilon=0.05,
            damping=0.8
        )
        jax.block_until_ready(u)
        elapsed = time.time() - start
        
        # Check drift
        max_drift = 0.0
        for t in range(N):
            drift = ops.compute_martingale_violation(
                u[t], u[t+1], h[t], x_grid, C, Delta, 0.05
            )
            max_drift = max(max_drift, float(drift))
        
        # Calculate metrics
        ops_per_iter = N * M * M  # Rough estimate of matrix ops per iteration
        total_ops = int(iters) * ops_per_iter
        throughput = total_ops / elapsed / 1e6  # Million ops/sec
        
        print(f"  Iterations: {int(iters)}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Time/iter: {elapsed/int(iters)*1000:.2f}ms")
        print(f"  Drift: {max_drift:.2e}")
        print(f"  Throughput: {throughput:.1f} MOps/sec")
        
        benchmarks.append({
            'label': label,
            'N': N,
            'M': M,
            'iters': int(iters),
            'time': elapsed,
            'drift': max_drift,
            'throughput': throughput
        })
    
    # Summary table
    print("\n" + "="*70)
    print("BENCHMARK SUMMARY")
    print("="*70)
    print(f"\nBackend: {backend.upper()}")
    print(f"\n{'Size':<12} {'N':<5} {'M':<5} {'Iters':<8} {'Time':<10} {'Throughput':<15} {'Drift'}")
    print("-"*70)
    for b in benchmarks:
        print(f"{b['label']:<12} {b['N']:<5} {b['M']:<5} {b['iters']:<8} {b['time']:<10.3f} {b['throughput']:<15.1f} {b['drift']:.2e}")
    
    # Performance assessment
    print("\n" + "="*70)
    print("PERFORMANCE ASSESSMENT")
    print("="*70)
    
    # Find N=50 benchmark
    xl_bench = next((b for b in benchmarks if b['N'] == 50), None)
    if xl_bench:
        if xl_bench['time'] < 5.0:
            print("✅ EXCELLENT: N=50 solved in < 5s")
        elif xl_bench['time'] < 15.0:
            print("✅ GOOD: N=50 solved in < 15s")
        else:
            print("⚠️  SLOW: N=50 took > 15s")
        
        if xl_bench['drift'] < 1e-4:
            print("✅ EXCELLENT: Drift < 1e-4")
        elif xl_bench['drift'] < 1e-2:
            print("✅ GOOD: Drift < 1e-2")
    
    if backend == 'cpu':
        print("\n💡 TIP: Running on CPU. For GPU acceleration:")
        print("   CUDA: pip install jax[cuda12]")
        print("   Metal: pip install jax-metal")
    
    return benchmarks

def main():
    """Run full profiling suite."""
    backend = check_jax_configuration()
    benchmarks = run_performance_benchmark()
    
    print("\n" + "="*70)
    print("🏁 PROFILING COMPLETE")
    print("="*70)
    
    return backend, benchmarks

if __name__ == "__main__":
    main()
