"""
STREAM-style memory bandwidth probe — JAX CPU, MLX GPU, MLX CPU.
Runs N_TRIALS independent trials and reports mean ± std.
"""

import time
import os
import statistics

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
import mlx.core as mx

SIZE_MB   = 512
N_TRIALS  = 5


def measure_bw(name, fn_array, fn_eval, size_mb=SIZE_MB, n_warmup=10):
    n = int(size_mb * 1e6 / 4)  # float32 elements

    a = fn_array(n)
    b = fn_array(n)
    fn_eval(a)
    fn_eval(b)

    for _ in range(n_warmup):
        b = a * 2.0
        fn_eval(b)

    t0 = time.perf_counter()
    for _ in range(5):
        b = a * 2.0
        fn_eval(b)
    t1 = time.perf_counter()

    return (5 * 2 * size_mb) / ((t1 - t0) * 1e3)  # GB/s


def run_trials(name, fn_array, fn_eval):
    results = []
    for i in range(N_TRIALS):
        bw = measure_bw(name, fn_array, fn_eval)
        results.append(bw)
        print(f"    trial {i+1}: {bw:.1f} GB/s")
    mean = statistics.mean(results)
    stdev = statistics.stdev(results)
    print(f"  {name:<12}  mean={mean:.1f}  std={stdev:.1f}  GB/s  (N={N_TRIALS})\n")
    return mean, stdev


print("=" * 55)
print(f"  STREAM probe  ({SIZE_MB} MB float32, 5 timed passes, {N_TRIALS} trials)")
print("=" * 55)

print("\n[JAX CPU]")
jax_mean, jax_std = run_trials(
    "JAX CPU",
    lambda n: jnp.ones(n, dtype=jnp.float32),
    lambda x: x.block_until_ready(),
)

mx.set_default_device(mx.gpu)
print("[MLX GPU]")
gpu_mean, gpu_std = run_trials(
    "MLX GPU",
    lambda n: mx.ones(n, dtype=mx.float32),
    lambda x: mx.eval(x),
)

mx.set_default_device(mx.cpu)
print("[MLX CPU]")
cpu_mean, cpu_std = run_trials(
    "MLX CPU",
    lambda n: mx.ones(n, dtype=mx.float32),
    lambda x: mx.eval(x),
)

print("=" * 55)
print("  SUMMARY")
print("=" * 55)
print(f"  JAX CPU : {jax_mean:.1f} ± {jax_std:.1f} GB/s")
print(f"  MLX GPU : {gpu_mean:.1f} ± {gpu_std:.1f} GB/s")
print(f"  MLX CPU : {cpu_mean:.1f} ± {cpu_std:.1f} GB/s")
