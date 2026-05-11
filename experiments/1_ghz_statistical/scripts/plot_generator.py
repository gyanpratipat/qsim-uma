"""
Quantum Simulation Memory Wall — Paper-Ready Plot Generation
Generates all figures needed for Medium article and academic paper.

Figures:
  Fig 1: Memory wall — state vector size vs qubit count (theory)
  Fig 2: Algorithm comparison — time vs qubits (4 algorithms, error bars)
  Fig 3: Memory footprint comparison — peak GB vs qubits
  Fig 4: PARAM Buddha CPU vs GPU (2022 historical data)
  Fig 5: M4 unified memory — GPU speedup vs qubit count
  Fig 6: The 29-qubit cliff — zoomed timing discontinuity
  Fig 7: Arithmetic intensity roofline diagram
  Fig 8: Memory hierarchy + qubit regime mapping
  Fig 9: Algorithm evolution timeline (story diagram)
  Fig 10: Storage-centric scaling (SSD experiment)
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import json
import glob
from pathlib import Path

# ── Style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'DejaVu Sans',
    'font.size':        11,
    'axes.titlesize':   13,
    'axes.labelsize':   12,
    'xtick.labelsize':  10,
    'ytick.labelsize':  10,
    'legend.fontsize':  10,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.3,
    'lines.linewidth':  2,
    'lines.markersize': 7,
})

COLORS = {
    'brute':           '#E74C3C',   # red — the bad approach
    'pykron':          '#F39C12',   # orange — intermediate
    'jax':             '#2ECC71',   # green — optimised
    'direct_index':            '#3498DB',   # blue — direct-index reconstruction (NumPy)
    'ssd':             '#9B59B6',   # purple — storage-centric
    'cpu':             '#7F8C8D',   # grey — CPU
    'gpu':             '#E67E22',   # amber — GPU
    'theory':          '#BDC3C7',   # light grey — theoretical
    'mlx_gpu_tensor':  '#E67E22',   # amber — MLX GPU tensor
    'mlx_gpu_flat':    '#F0B429',   # yellow-amber — MLX GPU flat
    'mlx_cpu_tensor':  '#1ABC9C',   # teal — MLX CPU tensor
    'mlx_cpu_flat':    '#76D7C4',   # light teal — MLX CPU flat
    'mlx_gpu_direct_index':    '#2980B9',   # dark blue — MLX GPU direct-index
    'mlx_cpu_direct_index':    '#85C1E9',   # light blue — MLX CPU direct-index
}

OUTDIR = Path(__file__).parent.parent / "results" / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

def save(fig, name):
    path = OUTDIR / f"{name}.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 1: Memory Wall Theory ─────────────────────────────────
def fig_memory_wall():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    q = np.arange(1, 42)
    sv_bytes = 2.0**q * 8  # complex64 = 8 bytes
    sv_gb    = sv_bytes / 1024**3
    
    ax.semilogy(q, sv_gb, color=COLORS['theory'], lw=2, label='State vector (O(2ⁿ))')
    ax.fill_between(q, sv_gb, alpha=0.1, color=COLORS['theory'])
    
    # Memory tier lines
    tiers = [
        (192e3/1024**3,   'L1 Cache (192 KB)',  '#E74C3C'),
        (16e6/1024**3,    'L2 Cache (16 MB)',   '#E67E22'),
        (24e6/1024**3,    'L3 Cache (24 MB)',   '#F1C40F'),
        (48.0,            'DRAM (48 GB)',        '#2ECC71'),
        (2000.0,          'NVMe SSD (~2 TB)',    '#3498DB'),
    ]
    for val, label, color in tiers:
        ax.axhline(val, color=color, linestyle='--', alpha=0.7, lw=1.5)
        ax.text(42, val, f' {label}', va='center', color=color, fontsize=9)
    
    # Your 2022 wall
    ax.axvline(14, color=COLORS['brute'], linestyle=':', lw=2, alpha=0.8)
    ax.text(14.2, 0.001, '14q: your\n2022 crash', color=COLORS['brute'], fontsize=9)
    
    # M4 wall
    ax.axvline(30, color=COLORS['jax'], linestyle=':', lw=2, alpha=0.8)
    ax.text(30.2, 0.001, '30q: M4\n2025 limit', color=COLORS['jax'], fontsize=9)
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('State Vector Size (GB)')
    ax.set_title('The Memory Wall: Exponential State Vector Growth\n'
                 'Every qubit added doubles memory requirements')
    ax.set_xlim(1, 41)
    ax.set_xticks(range(5, 41, 5))
    ax.legend(loc='upper left')
    
    fig.tight_layout()
    save(fig, "fig1_memory_wall_theory")


# ── Fig 2: Algorithm Time Comparison ─────────────────────────
def fig_algorithm_comparison(results):
    """Time vs qubits for all algorithms, with error bars."""
    fig, ax = plt.subplots(figsize=(12, 7))
    
    algo_map = {
        'A_BruteForce_NumPy':    ('Brute Force NumPy O(4ⁿ)',     COLORS['brute'],           'o-'),
        'B_pykronecker':         ('pykronecker O(2ⁿ)',            COLORS['pykron'],          's-'),
        'C_JAX_tensordot':       ('JAX CPU tensordot',            COLORS['jax'],             '^-'),
        'D_direct-index_Reconstruction': ('direct-index Reconstruction (NumPy)',  COLORS['direct_index'],            'D-'),
        'E_External_SSD':        ('Storage-Centric (NVMe)',        COLORS['ssd'],             'P-'),
        'F_MLX_GPU_Tensor':      ('MLX GPU tensor [2]*n',         COLORS['mlx_gpu_tensor'],  'o-'),
        'G_MLX_CPU_Tensor':      ('MLX CPU tensor [2]*n',         COLORS['mlx_cpu_tensor'],  's-'),
        'H_MLX_GPU_Flat':        ('MLX GPU flat 2^n',             COLORS['mlx_gpu_flat'],    '^--'),
        'I_MLX_CPU_Flat':        ('MLX CPU flat 2^n',             COLORS['mlx_cpu_flat'],    'D--'),
        'J_MLX_GPU_direct-index':        ('MLX GPU direct-index (Metal)',         COLORS['mlx_gpu_direct_index'],    'v-'),
        'K_MLX_CPU_direct-index':        ('MLX CPU direct-index',                 COLORS['mlx_cpu_direct_index'],    'v--'),
    }
    
    for algo_key, (label, color, style) in algo_map.items():
        data = [r for r in results if r['algorithm'] == algo_key]
        if not data:
            continue
        q   = [r['qubits']     for r in data]
        t   = [r['time_mean']  for r in data]
        err = [r['time_std']   for r in data]
        ax.errorbar(q, t, yerr=err, fmt=style, color=color,
                    label=label, capsize=4, capthick=1.5, alpha=0.9)
    
    # Annotate the 14q wall
    ax.axvline(14, color='grey', linestyle=':', alpha=0.5)
    ax.text(14.1, ax.get_ylim()[1]*0.8, '14q\n2022 crash',
            fontsize=8, color='grey')
    
    # Memory tier shading
    ax.axvspan(0, 14,  alpha=0.03, color='red',   label='L1 cache-resident')
    ax.axvspan(14, 21, alpha=0.03, color='orange', label='L2/L3 territory')
    ax.axvspan(21, 30, alpha=0.03, color='green',  label='DRAM-bound')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Wall-Clock Time (seconds)')
    ax.set_title(f'GHZ Circuit Simulation: Algorithm Comparison\n'
                 f'Apple M4 Pro, 48 GB Unified Memory '
                 f'(N={results[0]["n_trials"] if results else 7} trials, error bars = 1σ)')
    ax.set_yscale('log')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(2, 31)
    
    fig.tight_layout()
    save(fig, "fig2_algorithm_time_comparison")


# ── Fig 3: Memory Footprint Comparison ───────────────────────
def fig_memory_footprint(results):
    """Peak memory vs qubits for each algorithm."""
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Theoretical state vector
    q_range = np.arange(3, 32)
    sv_gb = 2.0**q_range * 8 / 1024**3
    ax.semilogy(q_range, sv_gb, color=COLORS['theory'], lw=1.5,
                linestyle='--', label='Theoretical O(2ⁿ) minimum')
    
    algo_map = {
        'A_BruteForce_NumPy':    ('Brute Force NumPy', COLORS['brute'],  'o-'),
        'B_pykronecker':         ('pykronecker',        COLORS['pykron'], 's-'),
        'D_direct-index_Reconstruction': ('direct-index Reconstruction',COLORS['direct_index'],   'D-'),
    }
    
    for algo_key, (label, color, style) in algo_map.items():
        data = [r for r in results if r['algorithm'] == algo_key]
        if not data:
            continue
        q = [r['qubits']     for r in data]
        m = [r['mem_max_gb'] for r in data]
        ax.semilogy(q, m, style, color=color, label=label)
    
    # JAX note
    ax.text(20, 1e-4, 'JAX: tracemalloc cannot\ncapture device allocator\n'
                      '(actual = O(2ⁿ) same as direct-index)',
            fontsize=8, style='italic', color=COLORS['jax'],
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Peak Memory Allocation (GB)')
    ax.set_title('Memory Footprint by Algorithm\n'
                 'Brute force uses up to 50,000× more memory than the state vector requires')
    ax.legend(loc='upper left')
    ax.set_xlim(2, 31)
    
    fig.tight_layout()
    save(fig, "fig3_memory_footprint")


# ── Fig 4: PARAM Buddha Historical (2022) ────────────────────
def fig_param_buddha():
    """Your 2022 PARAM Buddha data — the GPU irrelevance finding."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Data from your QRNG notebook — PARAM Buddha CPU vs GPU
    q = [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16]
    cpu_time = [0.00171, 0.00313, 0.00284, 0.00289, 0.00472, 0.00854,
                0.01708, 0.06646, 0.15204, 0.10409, 0.54132, 1.92557,
                7.13022, 28.63415, 119.70568, 503.95792]
    gpu_time = [1.42511, 0.00133, 0.00177, 0.00291, 0.00470, 0.01874,
                0.03301, 0.07013, 0.15702, 0.09548, 0.51891, 1.88272,
                6.96880, 28.78114, 120.75394, 497.56786]
    cpu_mem  = [1.25e-5, 7.71e-6, 9.58e-6, 1.60e-5, 5.35e-5, 2.05e-4,
                6.49e-4, 2.21e-3, 8.44e-3, 3.33e-2, 1.33e-1, 5.31e-1,
                2.125,   8.500,  34.001, 136.001]
    gpu_mem  = [7.01e-3, 7.16e-6, 7.78e-6, 1.52e-5, 5.20e-5, 2.02e-4,
                6.19e-4, 2.09e-3, 7.95e-3, 3.14e-2, 1.25e-1, 5.00e-1,
                2.000,   8.000,  32.001, 128.001]
    
    # Time plot
    ax = axes[0]
    ax.semilogy(q, cpu_time, 'o-', color=COLORS['cpu'], label='CPU (Intel Xeon)')
    ax.semilogy(q, gpu_time, 's-', color=COLORS['gpu'], label='GPU (NVIDIA, PARAM Buddha)')
    ax.axvline(1, color='red', linestyle=':', alpha=0.5)
    ax.text(1.1, 10, '1q: GPU 833× slower\n(init overhead)', fontsize=8, color='red')
    ax.axvline(16, color='grey', linestyle=':', alpha=0.5)
    ax.text(13.5, 0.0001, '16q: 1.3% GPU\nadvantage', fontsize=8, color='grey')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('PARAM Buddha 2022: CPU vs GPU\nBrute Force Algorithm (QRNG Circuit)')
    ax.legend()
    
    # Memory plot
    ax = axes[1]
    ax.semilogy(q, cpu_mem, 'o-', color=COLORS['cpu'], label='CPU peak memory')
    ax.semilogy(q, gpu_mem, 's-', color=COLORS['gpu'], label='GPU peak memory')
    q_arr = np.array(q, dtype=float)
    sv_theory = 2.0**q_arr * 8 / 1024**3
    ax.semilogy(q, sv_theory, '--', color=COLORS['theory'], alpha=0.5, label='O(2ⁿ) minimum')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Peak Memory (GB, log scale)')
    ax.set_title('Memory Consumption: CPU vs GPU\nBoth follow same O(4ⁿ) brute-force curve')
    ax.legend()
    
    fig.suptitle('PARAM Buddha National Supercomputer (2022)\n'
                 'GPU provides only 1.3% speedup at 16 qubits — brute force algorithm is the bottleneck',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig4_param_buddha_2022")


# ── Fig 5: M4 Unified Memory GPU Speedup ─────────────────────
def fig_m4_gpu_speedup(results):
    """Show how GPU speedup evolves with qubit count on M4, using benchmark stats."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    def extract(algo_key):
        data = sorted([r for r in results if r['algorithm'] == algo_key],
                      key=lambda r: r['qubits'])
        return data

    backends = [
        ('C_JAX_tensordot',  'JAX CPU (XLA/AMX)',  COLORS['jax'],           'o-'),
        ('F_MLX_GPU_Tensor', 'MLX GPU tensor',      COLORS['mlx_gpu_tensor'],'s-'),
        ('G_MLX_CPU_Tensor', 'MLX CPU tensor',      COLORS['mlx_cpu_tensor'],'^-'),
        ('J_MLX_GPU_direct-index',   'MLX GPU direct-index',        COLORS['mlx_gpu_direct_index'],  'v-'),
        ('K_MLX_CPU_direct-index',   'MLX CPU direct-index',        COLORS['mlx_cpu_direct_index'],  'v--'),
    ]

    ax = axes[0]
    backend_series = {}
    for algo_key, label, color, style in backends:
        data = extract(algo_key)
        if not data:
            continue
        q   = [r['qubits']    for r in data]
        t   = [r['time_mean'] for r in data]
        err = [r['time_std']  for r in data]
        ax.errorbar(q, t, yerr=err, fmt=style, color=color,
                    label=label, capsize=3, capthick=1.2, alpha=0.9)
        backend_series[algo_key] = {r['qubits']: r for r in data}

    # Annotate 29q cliff using actual data
    jax_data = backend_series.get('C_JAX_tensordot', {})
    if 28 in jax_data and 29 in jax_data:
        t28, t29 = jax_data[28]['time_mean'], jax_data[29]['time_mean']
        ratio = t29 / t28
        ax.annotate(f'29q DRAM cliff\n{ratio:.1f}× jump on JAX CPU',
                    xy=(29, t29), xytext=(26, t29* 0.4),
                    arrowprops=dict(arrowstyle='->', color='black'),
                    fontsize=9, color='black')

    ax.set_yscale('log')
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title(f'Apple M4 Pro: Three Backends\nUnified Memory Architecture '
                 f'(N={results[0]["n_trials"] if results else 7} trials, error bars = 1σ)')
    ax.legend()

    # Speedup plot: MLX GPU / JAX CPU
    ax = axes[1]
    jax_map = backend_series.get('C_JAX_tensordot', {})
    gpu_map = backend_series.get('F_MLX_GPU_Tensor', {})
    shared_q = sorted(set(jax_map) & set(gpu_map))
    if shared_q:
        speedups, speedup_errs, q_valid = [], [], []
        for q in shared_q:
            j, g = jax_map[q], gpu_map[q]
            if g['time_mean'] > 0:
                s = j['time_mean'] / g['time_mean']
                # error propagation: σ_s/s = sqrt((σ_j/μ_j)² + (σ_g/μ_g)²)
                rel_err = np.sqrt((j['time_std']/j['time_mean'])**2 +
                                  (g['time_std']/g['time_mean'])**2)
                speedups.append(s)
                speedup_errs.append(s * rel_err)
                q_valid.append(q)
        ax.errorbar(q_valid, speedups, yerr=speedup_errs, fmt='D-',
                    color=COLORS['mlx_gpu_tensor'], capsize=3,
                    label='MLX GPU / JAX CPU speedup')

    ax.axhline(1, color='grey', linestyle='--', alpha=0.5, label='No speedup (1×)')
    ax.scatter([32], [18], marker='*', s=200, color='red', zorder=5,
               label='QARN (NVIDIA L40, dedicated VRAM)')
    ax.annotate('QARN: 18×\n(12× BW ratio)', xy=(32, 18), xytext=(28, 15),
                arrowprops=dict(arrowstyle='->', color='red'), fontsize=9, color='red')
    ax.text(3, 2.5, 'M4 peak: ~3×\n(1.44× BW ratio)\nUnified memory\nno PCIe overhead',
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('GPU Speedup (×)')
    ax.set_title('GPU Speedup Bounded by Memory Bandwidth Ratio\n'
                 'M4 unified memory vs NVIDIA dedicated VRAM')
    ax.legend(loc='upper right')

    fig.suptitle('Unified Memory Architecture Makes the Memory Bandwidth Theory Visible\n'
                 'GPU speedup ≈ bandwidth ratio, not compute ratio',
                 fontsize=12, y=1.02)
    fig.tight_layout()
    save(fig, "fig5_m4_unified_memory_speedup")


# ── Fig 6: The 29-Qubit Cliff ─────────────────────────────────
def fig_29q_cliff(results):
    """Zoom in on the 29-qubit DRAM bandwidth cliff using benchmark statistics."""
    fig, ax = plt.subplots(figsize=(10, 6))

    backends = [
        ('C_JAX_tensordot',  'JAX CPU',  COLORS['jax'],           'o-', 2.5),
        ('F_MLX_GPU_Tensor', 'MLX GPU',  COLORS['mlx_gpu_tensor'],'s-', 2.5),
        ('G_MLX_CPU_Tensor', 'MLX CPU',  COLORS['mlx_cpu_tensor'],'^-', 2.5),
    ]

    series = {}
    for algo_key, label, color, style, lw in backends:
        data = sorted([r for r in results
                       if r['algorithm'] == algo_key and 24 <= r['qubits'] <= 31],
                      key=lambda r: r['qubits'])
        if not data:
            continue
        q   = [r['qubits']    for r in data]
        t   = [r['time_mean'] for r in data]
        err = [r['time_std']  for r in data]
        ax.errorbar(q, t, yerr=err, fmt=style, color=color, lw=lw,
                    label=label, capsize=4, capthick=1.5)
        series[algo_key] = {r['qubits']: r for r in data}

    ax.axvspan(28.5, 29.5, alpha=0.15, color='red', label='DRAM bandwidth cliff')

    # Dynamic annotations from actual data
    jax_s = series.get('C_JAX_tensordot', {})
    gpu_s = series.get('F_MLX_GPU_Tensor', {})
    if 28 in jax_s and 29 in jax_s:
        t28, t29 = jax_s[28]['time_mean'], jax_s[29]['time_mean']
        ratio = t29 / t28
        ax.annotate(
            f'28q→29q: JAX CPU\n{t28:.1f}s → {t29:.1f}s ({ratio:.1f}×)\n'
            f'State vector: 2.1GB → 4.3GB\nCrosses DRAM bandwidth ceiling',
            xy=(29, t29), xytext=(26.5, t29 * 1.1),
            arrowprops=dict(arrowstyle='->', color='red', lw=2),
            fontsize=10, color='red',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
    if 28 in gpu_s and 29 in gpu_s:
        t28g, t29g = gpu_s[28]['time_mean'], gpu_s[29]['time_mean']
        ratio_g = t29g / t28g
        ax.annotate(
            f'GPU cliff softer:\n{t28g:.1f}s → {t29g:.1f}s ({ratio_g:.1f}×)\n'
            f'Concurrent mem transactions\npartially compensate',
            xy=(29, t29g), xytext=(29.2, t29g * 0.5),
            arrowprops=dict(arrowstyle='->', color=COLORS['mlx_gpu_tensor'], lw=1.5),
            fontsize=9, color=COLORS['mlx_gpu_tensor'])

    # State vector size labels on x-axis
    for qq, sv in [(25, 0.268), (27, 1.074), (28, 2.147), (29, 4.295), (30, 8.590)]:
        ax.text(qq, ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.1,
                f'{sv:.1f}GB', fontsize=8, ha='center', color='grey')

    n_trials = results[0]['n_trials'] if results else 7
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title(f'The 29-Qubit DRAM Bandwidth Cliff — A Phase Transition, Not a Smooth Curve\n'
                 f'(N={n_trials} trials, error bars = 1σ)')
    ax.legend(loc='upper left')
    ax.set_xlim(24.5, 31.5)
    ax.set_yscale('log')

    fig.tight_layout()
    save(fig, "fig6_29qubit_cliff")


# ── Fig 7: Roofline Diagram ───────────────────────────────────
def fig_roofline():
    """Roofline model showing all quantum gate ops are memory-bound."""
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # M4 Pro specs
    bw_cpu   = 106    # GB/s JAX CPU
    bw_gpu   = 137.5  # GB/s MLX GPU  
    flops_cpu= 3600   # GFLOP/s
    flops_gpu= 10000  # GFLOP/s (estimated Metal)
    
    ai_range = np.logspace(-3, 3, 1000)
    
    # Roofline ceilings — CPU
    roof_cpu = np.minimum(flops_cpu, ai_range * bw_cpu)
    ax.loglog(ai_range, roof_cpu, '-', color=COLORS['cpu'], lw=2, 
              label=f'JAX CPU (BW={bw_cpu}GB/s, Peak={flops_cpu}GFLOP/s)')
    
    # Roofline ceilings — GPU
    roof_gpu = np.minimum(flops_gpu, ai_range * bw_gpu)
    ax.loglog(ai_range, roof_gpu, '-', color=COLORS['gpu'], lw=2,
              label=f'MLX GPU (BW={bw_gpu}GB/s, Peak={flops_gpu}GFLOP/s)')
    
    # Ridge points
    ridge_cpu = flops_cpu / bw_cpu  # ~34 FLOP/byte
    ridge_gpu = flops_gpu / bw_gpu  # ~73 FLOP/byte
    ax.axvline(ridge_cpu, color=COLORS['cpu'], linestyle=':', alpha=0.5)
    ax.axvline(ridge_gpu, color=COLORS['gpu'], linestyle=':', alpha=0.5)
    ax.text(ridge_cpu*1.1, 100, f'CPU ridge\n{ridge_cpu:.0f} F/B', 
            fontsize=8, color=COLORS['cpu'])
    ax.text(ridge_gpu*1.1, 100, f'GPU ridge\n{ridge_gpu:.0f} F/B', 
            fontsize=8, color=COLORS['gpu'])
    
    # Gate operations (CNOT has 0 FLOP so is shown at left edge, not log-zero)
    gate_ops = [
        (6/32,   5,    'Hadamard (6 FLOP, 32B)',    COLORS['direct_index']),
        (1e-3,   1,    'CNOT/X (≈0 FLOP, 32B)',     COLORS['brute']),   # clipped to xlim left
        (8/32,   20,   'Brute force matmul',          COLORS['pykron']),
    ]
    for ai, perf, label, color in gate_ops:
        ax.scatter([ai], [perf], s=150, color=color, zorder=5)
        ax.text(ai * 1.5, perf * 1.5, label, fontsize=9, color=color)
    
    # Memory-bound shading
    ax.axvspan(1e-3, ridge_cpu, alpha=0.05, color='red')
    ax.text(1e-2, 10, 'MEMORY\nBOUND', fontsize=12, color='red', 
            alpha=0.4, weight='bold')
    ax.text(ridge_cpu*5, 1000, 'COMPUTE\nBOUND', fontsize=12, 
            color='green', alpha=0.4, weight='bold')
    
    ax.set_xlabel('Arithmetic Intensity (FLOP/byte)')
    ax.set_ylabel('Performance (GFLOP/s)')
    ax.set_title('Roofline Model: All Quantum Gates Are Memory-Bound\n'
                 'Apple M4 Pro — JAX CPU vs MLX GPU')
    ax.legend(loc='upper left')
    ax.set_xlim(1e-3, 1e3)
    ax.set_ylim(1, 1e5)
    
    fig.tight_layout()
    save(fig, "fig7_roofline_model")


# ── Fig 8: Memory Hierarchy Map ──────────────────────────────
def fig_memory_hierarchy_map():
    """Visual map of qubit regimes vs memory hierarchy."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    
    # Hierarchy bands
    bands = [
        (0,  14, '#FFE6E6', 'L1/L2 Cache\n≤14q\nCompute-bound\n(3–14 qubits)'),
        (14, 21, '#FFF3E6', 'L2/L3 Cache\n14–21q\nCache BW-bound'),
        (21, 29, '#E6F3FF', 'DRAM\n21–29q\nBandwidth-bound'),
        (29, 31, '#FFE6FF', '29q Cliff\nPhase transition'),
        (31, 37, '#E6FFE6', 'NVMe SSD\n31–37q\nStorage-bound'),
        (37, 40, '#F5F5F5', '>37q\nDistributed\nonly'),
    ]
    for x0, x1, color, label in bands:
        ax.axvspan(x0, x1, alpha=0.4, color=color)
        ax.text((x0+x1)/2, 0.5, label, ha='center', va='center', 
                fontsize=9, wrap=True)
    
    # Your data points
    events = [
        (3,  'V1 start',     'o', COLORS['brute']),
        (14, '2022 crash',   '*', COLORS['brute']),
        (19, 'JAX on\nPARAM', 'D', COLORS['jax']),
        (30, 'M4 direct-index\nlimit','s', COLORS['direct_index']),
        (32, 'QARN\n(2025)', '^', '#FF0000'),
    ]
    for q, label, marker, color in events:
        ax.scatter([q], [0.85], marker=marker, s=150, color=color, zorder=5)
        ax.text(q, 0.92, label, ha='center', fontsize=8, color=color)
    
    # Bandwidth bars below
    bw_data = [
        (0,  14, 3000, 'L1: ~3000 GB/s'),
        (14, 21, 2000, 'L2: ~2000 GB/s'),
        (21, 29, 106,  'DRAM: ~106 GB/s'),
        (31, 37, 3.5,  'NVMe: ~3.5 GB/s'),
    ]
    for x0, x1, bw, label in bw_data:
        bar_h = min(0.25, np.log10(bw+1) * 0.08)
        ax.barh(0.12, x1-x0, left=x0, height=bar_h, 
                color='steelblue', alpha=0.6, align='edge')
        ax.text((x0+x1)/2, 0.05, label, ha='center', fontsize=7, color='navy')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_title('Quantum Simulation Memory Hierarchy Map — Apple M4 Pro\n'
                 'Each regime has a different dominant bottleneck',
                 fontsize=13)
    
    # Legend
    legend_elements = [
        mpatches.Patch(color=COLORS['brute'], alpha=0.7, label='Your 2022 work'),
        mpatches.Patch(color=COLORS['jax'],   alpha=0.7, label='JAX optimization'),
        mpatches.Patch(color=COLORS['direct_index'],  alpha=0.7, label='direct-index/M4 2025'),
        mpatches.Patch(color='red',           alpha=0.7, label='QARN paper'),
    ]
    ax.legend(handles=legend_elements, loc='lower right')
    
    fig.tight_layout()
    save(fig, "fig8_memory_hierarchy_map")


# ── Fig 9: Evolution Timeline ─────────────────────────────────
def fig_evolution_timeline():
    """Your simulator evolution as a story diagram."""
    fig, ax = plt.subplots(figsize=(14, 7))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.5, 5)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.axis('off')
    
    stages = [
        # x, y, title, subtitle, color
        (0.5, 4.0, 'V1', 'Single-qubit gates\nnp.matmul(gate, state)\nO(2n) — correct', COLORS['brute']),
        (2.5, 4.0, 'V2', 'Multi-qubit\nnp.kron(I, gate, I)\nO(4^n) — the mistake', COLORS['brute']),
        (4.5, 4.0, 'Wall', '14-qubit crash\n61.9s / 12.1 GB\nGPU: 1.3% faster', '#E74C3C'),
        (6.5, 4.0, 'V3a', 'pykronecker\nLazy Kronecker\nO(2^n × k)', COLORS['pykron']),
        (8.5, 4.0, 'V3b', 'JAX+CuPy+pykron\nXLA fusion\nPARAM Buddha', COLORS['jax']),
        
        # Second row — applications and validation
        (1.5, 2.0, 'GHZ\nBenchmark', 'PARAM Buddha\nCPU vs GPU\n2022', '#95A5A6'),
        (3.5, 2.0, 'QRNG', 'Quantum RNG\nFunctional\nproof', '#95A5A6'),
        (5.5, 2.0, 'Reversibility\nTest', 'Bell → reverse\n|00⟩ confirmed\nUnitarity ✓', '#95A5A6'),
        (7.5, 2.0, 'Grover\n14q', 'Probability\n0.9999998\nCorrectness ✓', '#95A5A6'),
        
        # Bottom row — 2025
        (3.5, 0.2, 'QARN\n(2025)', 'Rejeesh + Nishant\nARUN algorithm\nIEEE SCI 2025', '#E74C3C'),
        (7.0, 0.2, 'M4\nAnalysis', 'pykron / JAX / direct-index\n3× GPU, 29q cliff\nUnified memory', COLORS['direct_index']),
    ]
    
    for x, y, title, subtitle, color in stages:
        # Box
        rect = mpatches.FancyBboxPatch((x-0.4, y-0.4), 0.8, 0.8, 
                                        boxstyle='round,pad=0.1',
                                        facecolor=color, alpha=0.3, edgecolor=color, lw=2)
        ax.add_patch(rect)
        ax.text(x, y+0.1, title, ha='center', va='center', fontsize=9, 
                weight='bold', color=color)
        ax.text(x, y-0.6, subtitle, ha='center', va='top', fontsize=7, 
                color='#555555', style='italic')
    
    # Arrows between stages
    arrow_props = dict(arrowstyle='->', lw=1.5, color='grey')
    for (x1, y1), (x2, y2) in [
        ((1.0, 4.0), (2.0, 4.0)),
        ((3.0, 4.0), (4.0, 4.0)),
        ((5.0, 4.0), (6.0, 4.0)),
        ((7.0, 4.0), (8.0, 4.0)),
    ]:
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=arrow_props)
    
    # Year labels
    for x, year in [(1, '2023'), (3, '2023'), (5, '2023–24'), (7, '2024'), (9, '2024')]:
        ax.text(x, 3.4, year, ha='center', fontsize=8, color='grey')
    for x, year in [(3.5, '2025'), (7.0, '2025')]:
        ax.text(x, -0.4, year, ha='center', fontsize=8, color='grey')
    
    ax.set_title('Simulator Evolution: From Single-Qubit Sketch to Research Platform\n'
                 '2023–2025', fontsize=13, pad=20)
    
    fig.tight_layout()
    save(fig, "fig9_evolution_timeline")


# ── Fig 10: Storage-Centric Scaling ──────────────────────────
def fig_ssd_scaling(results):
    """Show storage-centric simulation extending beyond DRAM wall."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    ssd_data = [r for r in results if r['algorithm'] == 'E_External_SSD']
    if ssd_data:
        q = [r['qubits']    for r in ssd_data]
        t = [r['time_mean'] for r in ssd_data]
        e = [r['time_std']  for r in ssd_data]
        ax.errorbar(q, t, yerr=e, fmt='P-', color=COLORS['ssd'], 
                    label='Storage-centric (NVMe memmap)', capsize=4)
    
    # Comparison: what DRAM-bound direct-index would cost (extrapolated)
    q_range = np.array([28, 29, 30, 31, 32, 33])
    # Rough extrapolation from M4 direct-index timing: doubles every qubit
    direct_index_extrap = np.array([10.1, 49.7, 136.5, 300, 700, 1600])
    ax.semilogy(q_range, direct_index_extrap, '--', color=COLORS['direct_index'], 
                alpha=0.5, label='direct-index (DRAM, extrapolated for 31q+)')
    
    ax.axvline(30, color='red', linestyle=':', alpha=0.5)
    ax.text(30.1, 1, '30q: M4 DRAM\nwall', fontsize=8, color='red')
    
    ax.axvspan(30, 40, alpha=0.05, color='purple', label='NVMe territory')
    
    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('Storage-Centric Simulation: Extending Beyond the DRAM Wall\n'
                 'NVMe bandwidth (~3.5 GB/s) is the new bottleneck')
    ax.legend()
    
    # Bandwidth comparison annotation
    ax.text(31, 10, 'NVMe BW: ~3.5 GB/s\nDRAM BW: ~106 GB/s\nSlowdown: ~30×\n(per access)', 
            fontsize=9, style='italic',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
    
    fig.tight_layout()
    save(fig, "fig10_ssd_scaling")


# ── Fig 11: Flat vs Tensor Representation ────────────────────
def fig_flat_vs_tensor(results):
    """
    Compare flat (2^n) vs tensor ([2]*n) MLX representation for GPU and CPU.
    Error bars from 7-trial statistics highlight the consistency of the gap.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [
        ('F_MLX_GPU_Tensor', 'H_MLX_GPU_Flat', 'MLX GPU (Metal)', axes[0]),
        ('G_MLX_CPU_Tensor', 'I_MLX_CPU_Flat', 'MLX CPU',         axes[1]),
    ]

    for tensor_key, flat_key, title, ax in pairs:
        tensor_data = sorted([r for r in results if r['algorithm'] == tensor_key],
                             key=lambda r: r['qubits'])
        flat_data   = sorted([r for r in results if r['algorithm'] == flat_key],
                             key=lambda r: r['qubits'])

        if tensor_data:
            q = [r['qubits']    for r in tensor_data]
            t = [r['time_mean'] for r in tensor_data]
            e = [r['time_std']  for r in tensor_data]
            ax.errorbar(q, t, yerr=e, fmt='s-', color='steelblue', capsize=4,
                        capthick=1.5, label='Tensor [2]*n', lw=2)

        if flat_data:
            q = [r['qubits']    for r in flat_data]
            t = [r['time_mean'] for r in flat_data]
            e = [r['time_std']  for r in flat_data]
            ax.errorbar(q, t, yerr=e, fmt='o--', color='coral', capsize=4,
                        capthick=1.5, label='Flat 2^n', lw=2)

        # Delta% annotation at shared qubit counts
        t_map = {r['qubits']: r['time_mean'] for r in tensor_data}
        f_map = {r['qubits']: r['time_mean'] for r in flat_data}
        for q in sorted(set(t_map) & set(f_map)):
            if q in (20, 24, 28):
                delta = (f_map[q] - t_map[q]) / t_map[q] * 100
                sign  = '+' if delta > 0 else ''
                ax.annotate(f'{sign}{delta:.0f}%',
                            xy=(q, max(t_map[q], f_map[q])),
                            xytext=(q + 0.3, max(t_map[q], f_map[q]) * 1.2),
                            fontsize=8, color='dimgrey')

        ax.set_yscale('log')
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Time (seconds, log scale)')
        ax.set_title(title)
        ax.legend()

    n_trials = results[0]['n_trials'] if results else 7
    fig.suptitle(
        f'Flat vs Tensor State Vector Representation — MLX GPU and CPU\n'
        f'Tensor [2]*n bypasses MLX int32 size ceiling; consistently faster on GPU\n'
        f'(N={n_trials} trials, error bars = 1σ)',
        fontsize=12)
    fig.tight_layout()
    save(fig, "fig11_flat_vs_tensor")


# ── Fig 12: GHZ Effective Bandwidth vs Qubit Count ───────────
def fig_ghz_eff_bandwidth(results):
    """Effective memory bandwidth per backend for the GHZ circuit.

    Formula: eff_bw = n_qubits * 2 * state_vector_gb / time_mean
    Counts n gates (1 H + n-1 CNOTs), each reading and writing the full
    state vector once (2 passes).  Headline result: MLX GPU direct-index (J)
    stays flat at ~45 GB/s straight through the 29q DRAM cliff while
    all tensordot backends drop ~50%.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Pre-compute eff_bw for every result row
    bw_map = {}   # (algorithm_key, qubits) -> eff_bw
    KEY_MAP = {
        'A_BruteForce_NumPy':    'A', 'B_pykronecker':         'B',
        'C_JAX_tensordot':       'C', 'D_direct-index_Reconstruction': 'D',
        'E_External_SSD':        'E', 'F_MLX_GPU_Tensor':      'F',
        'G_MLX_CPU_Tensor':      'G', 'H_MLX_GPU_Flat':        'H',
        'I_MLX_CPU_Flat':        'I', 'J_MLX_GPU_direct-index':        'J',
        'K_MLX_CPU_direct-index':        'K',
    }
    for r in results:
        key = KEY_MAP.get(r['algorithm'])
        if key and r['time_mean'] > 0:
            bw = r['qubits'] * 2 * r['state_vector_gb'] / r['time_mean']
            bw_map[(key, r['qubits'])] = bw

    def series(key):
        rows = sorted(
            [(q, bw) for (k, q), bw in bw_map.items() if k == key and q >= 15],
            key=lambda x: x[0])
        if not rows:
            return [], []
        return zip(*rows)

    LSTYLE = {'F': '-', 'H': '--', 'J': '-', 'C': '-', 'D': '-',
              'G': '-', 'I': '--', 'K': '--'}
    MARKER = {'F': 'o', 'H': '^', 'J': 'v', 'C': '^', 'D': 'D',
              'G': 's', 'I': 'D', 'K': 'v'}
    LABEL  = {
        'F': 'F: MLX GPU tensor',   'H': 'H: MLX GPU flat',
        'J': 'J: MLX GPU direct-index ★',  'C': 'C: JAX CPU (tensordot)',
        'D': 'D: NumPy direct-index',       'G': 'G: MLX CPU tensor',
        'I': 'I: MLX CPU flat',     'K': 'K: MLX CPU direct-index',
    }
    CLR = {
        'F': COLORS['mlx_gpu_tensor'], 'H': COLORS['mlx_gpu_flat'],
        'J': COLORS['mlx_gpu_direct_index'],   'C': COLORS['jax'],
        'D': COLORS['direct_index'],           'G': COLORS['mlx_cpu_tensor'],
        'I': COLORS['mlx_cpu_flat'],   'K': COLORS['mlx_cpu_direct_index'],
    }

    panels = [
        ('GPU Backends', ['F', 'H', 'J'], axes[0]),
        ('CPU Backends', ['C', 'D', 'G', 'K'], axes[1]),
    ]

    for panel_title, keys, ax in panels:
        for key in keys:
            qs, bws = series(key)
            if not qs:
                continue
            lw = 2.5 if key in ('J', 'K') else 1.8
            ax.plot(list(qs), list(bws),
                    LSTYLE[key], color=CLR[key],
                    marker=MARKER[key], lw=lw,
                    label=LABEL[key], markevery=3)

        # Cliff band
        ax.axvspan(28.5, 29.5, alpha=0.15, color='red', label='DRAM cliff')

        # State-size labels
        for qq, sv in [(25, '268 MB'), (27, '1.1 GB'),
                       (28, '2.1 GB'), (29, '4.3 GB'), (30, '8.6 GB')]:
            ybot = ax.get_ylim()[0] if ax.get_ylim()[0] > 0 else 0.5
            ax.text(qq, ybot, sv, fontsize=7, ha='center', color='grey')

        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Effective Bandwidth (GB/s)')
        ax.set_title(panel_title)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(14, 31)
        ax.set_xticks(range(15, 31, 2))

    # Re-draw state labels after y-limits are finalised
    for panel_title, keys, ax in panels:
        ybot = ax.get_ylim()[0] * 1.05 if ax.get_ylim()[0] > 0 else 0.3
        for qq, sv in [(25, '268 MB'), (27, '1.1 GB'),
                       (28, '2.1 GB'), (29, '4.3 GB'), (30, '8.6 GB')]:
            ax.text(qq, ybot, sv, fontsize=7, ha='center', color='grey')

    fig.suptitle(
        'GHZ Circuit: Effective Memory Bandwidth vs Qubit Count\n'
        'Formula: n_gates × 2 × state_GB / time  |  '
        'MLX GPU direct-index holds ~45 GB/s flat through the 29q DRAM cliff',
        fontsize=12)
    fig.tight_layout()
    save(fig, "fig12_ghz_eff_bandwidth")


# ── Main ──────────────────────────────────────────────────────
def _find_latest_inference():
    """Return path to the most recent inference/results.json, or None."""
    here = Path(__file__).parent.parent
    matches = sorted(glob.glob(str(here / "results" / "exp_*" / "inference" / "results.json")))
    if matches:
        return Path(matches[-1])
    return None


def main():
    print("\n── Generating Paper-Ready Figures ──────────────────────")

    results_path = _find_latest_inference()
    if results_path and results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"  Loaded {len(results)} benchmark results from {results_path}")
    else:
        print("  No benchmark results found — generating plots with example data")
        results = []
    
    print("\nGenerating figures:")
    fig_memory_wall()
    fig_algorithm_comparison(results)
    fig_memory_footprint(results)
    fig_param_buddha()
    fig_m4_gpu_speedup(results)
    fig_29q_cliff(results)
    fig_roofline()
    fig_memory_hierarchy_map()
    fig_evolution_timeline()
    fig_ssd_scaling(results)
    fig_flat_vs_tensor(results)
    fig_ghz_eff_bandwidth(results)

    print(f"\n  All figures saved to {OUTDIR}")
    print("  Format: PNG, 300 DPI (paper-ready)")


if __name__ == "__main__":
    main()
