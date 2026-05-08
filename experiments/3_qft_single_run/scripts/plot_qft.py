"""
QFT Benchmark — Paper-Ready Plot Generation
Parses qft_bench_console.log and generates figures for the QFT experiment.

Figures:
  Fig 1: All backends — time vs qubits (log scale)
  Fig 2: Effective bandwidth vs qubits (competitive backends)
  Fig 3: DRAM cliff zoom (24–30q)
  Fig 4: direct-index vs tensor — GPU and CPU side-by-side
  Fig 5: CPU double cliff — two cache transitions on MLX CPU
  Fig 6: 30q performance summary (bar chart)
  Fig 7: DRAM cliff — GHZ vs QFT side-by-side
  Fig 8: GHZ vs QFT performance comparison
"""

import re
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Style (matches plot_generator.py) ─────────────────────────
plt.rcParams.update({
    'font.family':       'DejaVu Sans',
    'font.size':         11,
    'axes.titlesize':    13,
    'axes.labelsize':    12,
    'xtick.labelsize':   10,
    'ytick.labelsize':   10,
    'legend.fontsize':   10,
    'figure.dpi':        150,
    'savefig.dpi':       300,
    'savefig.bbox':      'tight',
    'axes.spines.top':   False,
    'axes.spines.right': False,
    'axes.grid':         True,
    'grid.alpha':        0.3,
    'lines.linewidth':   2,
    'lines.markersize':  7,
})

COLORS = {
    'A': '#E74C3C',
    'B': '#F39C12',
    'C': '#2ECC71',
    'D': '#3498DB',
    'E': '#9B59B6',
    'F': '#E67E22',
    'G': '#1ABC9C',
    'H': '#F0B429',
    'I': '#76D7C4',
    'J': '#2980B9',
    'K': '#85C1E9',
}

MARKERS = {
    'A': 'o', 'B': 's', 'C': '^', 'D': 'D', 'E': 'P',
    'F': 'o', 'G': 's', 'H': '^', 'I': 'D', 'J': 'v', 'K': 'v',
}

LABELS = {
    'A': 'A: NumPy brute force (Kronecker)',
    'B': 'B: pykronecker',
    'C': 'C: JAX CPU (tensordot)',
    'D': 'D: NumPy direct-index (bitmask)',
    'E': 'E: SSD memmap direct-index',
    'F': 'F: MLX GPU tensor',
    'G': 'G: MLX CPU tensor',
    'H': 'H: MLX GPU flat',
    'I': 'I: MLX CPU flat',
    'J': 'J: MLX GPU direct-index ★',
    'K': 'K: MLX CPU direct-index',
}

LOG_PATH = Path(__file__).parent.parent / "logs" / "qft_bench_console.log"
OUTDIR   = Path(__file__).parent.parent / "results" / "figures_qft"
OUTDIR.mkdir(parents=True, exist_ok=True)


# ── Parser ─────────────────────────────────────────────────────
def parse_log(log_path=LOG_PATH):
    text      = Path(log_path).read_text()
    header_re = re.compile(r'^=== ([A-K])\s+.+? ===', re.MULTILINE)
    row_re    = re.compile(
        r'^\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)\s+([\d.]+)\s+(\w+)',
        re.MULTILINE)

    backends = {}
    headers  = list(header_re.finditer(text))
    for i, m in enumerate(headers):
        key = m.group(1)
        end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        rows = [
            {
                'qubits':   int(r.group(1)),
                'time':     float(r.group(2)),
                'n_gates':  int(r.group(3)),
                'state_gb': float(r.group(4)),
                'eff_gb_s': float(r.group(5)),
            }
            for r in row_re.finditer(text[m.end():end])
        ]
        backends[key] = rows
    return backends


def _xy(data, key='time'):
    return [r['qubits'] for r in data], [r[key] for r in data]


def save(fig, name):
    path = OUTDIR / f"{name}.png"
    fig.savefig(path)
    print(f"  Saved: {path}")
    plt.close(fig)


# ── Fig 1: All Backends — Time vs Qubits ─────────────────────
def fig_all_backends(bk):
    fig, ax = plt.subplots(figsize=(13, 7))

    linestyles = {'A': '-', 'B': '-', 'C': '-', 'D': '-', 'E': '-',
                  'F': '-', 'G': '-', 'H': '--', 'I': '--', 'J': '-', 'K': '--'}

    for key in 'ABCDEFGHIJK':
        if key not in bk or not bk[key]:
            continue
        q, t = _xy(bk[key])
        ax.semilogy(q, t, linestyles[key], color=COLORS[key],
                    marker=MARKERS[key], label=LABELS[key],
                    markevery=4, alpha=0.9)

    # DRAM cliff band
    ax.axvspan(28.5, 29.5, alpha=0.12, color='red', label='DRAM cliff region')

    # Memory tier annotations
    for q_line, label, c in [(21, '21q: pykron/CPU cliff', '#F39C12'),
                              (29, '29q: DRAM cliff',       '#E74C3C')]:
        ax.axvline(q_line, color=c, linestyle=':', lw=1.5, alpha=0.6)
        ax.text(q_line + 0.15, 2000, label, fontsize=8, color=c, rotation=90,
                va='top')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Wall-Clock Time (seconds, log scale)')
    ax.set_title('QFT Circuit Simulation: All 11 Backends\n'
                 'Apple M4 Pro, 48 GB Unified Memory — O(n²) gates')
    ax.legend(loc='upper left', ncol=2, framealpha=0.9)
    ax.set_xlim(3, 31)
    ax.set_xticks(range(3, 31, 2))
    fig.tight_layout()
    save(fig, "fig_qft1_all_backends")


# ── Fig 2: Effective Bandwidth ────────────────────────────────
def fig_eff_bandwidth(bk):
    """Two-panel GPU / CPU effective bandwidth, starting at 16q.

    Known JIT artifacts excluded:
      C@20q — JAX reused the warmup-compiled kernel (5.26 GB/s outlier)
      J@16q — Metal shader recompile (0.31 GB/s dip between 1.19 and 3.57)
    """
    JIT_SKIP = {('C', 20), ('J', 16)}

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)

    panels = [
        ('GPU Backends', ['F', 'H', 'J'], axes[0]),
        ('CPU Backends', ['C', 'D', 'G', 'K'], axes[1]),
    ]
    linestyles = {'F': '-', 'H': '--', 'J': '-',
                  'C': '-', 'D': '-', 'G': '-', 'K': '--'}

    for panel_title, keys, ax in panels:
        for key in keys:
            if key not in bk:
                continue
            rows = [r for r in bk[key]
                    if r['qubits'] >= 16 and (key, r['qubits']) not in JIT_SKIP]
            if not rows:
                continue
            qs = [r['qubits']   for r in rows]
            bs = [r['eff_gb_s'] for r in rows]
            ax.plot(qs, bs, linestyles[key], color=COLORS[key],
                    marker=MARKERS[key], label=LABELS[key], alpha=0.9)

        # Memory regime shading
        ax.axvspan(20.5, 21.5, alpha=0.10, color='#F39C12')
        ax.axvspan(28.5, 29.5, alpha=0.12, color='red')
        ax.text(21, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 14,
                '21q\nL3', ha='center', fontsize=7, color='#F39C12')
        ax.text(29, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 14,
                '29q\nDRAM', ha='center', fontsize=7, color='red')

        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Effective Bandwidth (GB/s)')
        ax.set_title(panel_title)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(15, 31)
        ax.set_xticks(range(16, 31, 2))

    # Re-apply shading labels after axes limits are set
    for panel_title, keys, ax in panels:
        ymax = ax.get_ylim()[1]
        ax.text(21, ymax * 0.95, '21q\nL3',   ha='center', fontsize=7, color='#F39C12')
        ax.text(29, ymax * 0.95, '29q\nDRAM', ha='center', fontsize=7, color='red')

    fig.suptitle('QFT Effective Bandwidth vs Qubit Count\n'
                 'Model: H gate = full state ×2 passes, CP = quarter state ×2, SWAP = quarter ×4',
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_qft2_eff_bandwidth")


# ── Fig 3: DRAM Cliff Zoom ────────────────────────────────────
def fig_cliff_zoom(bk):
    fig, ax = plt.subplots(figsize=(10, 6))

    show = ['C', 'D', 'F', 'G', 'H', 'J', 'K']
    linestyles = {'C': '-', 'D': '-', 'F': '-', 'G': '-',
                  'H': '--', 'J': '-', 'K': '--'}

    series = {}
    for key in show:
        if key not in bk:
            continue
        rows = [r for r in bk[key] if 24 <= r['qubits'] <= 30]
        if not rows:
            continue
        q, t = _xy(rows)
        ax.semilogy(q, t, linestyles[key], color=COLORS[key],
                    marker=MARKERS[key], label=LABELS[key])
        series[key] = {r['qubits']: r['time'] for r in rows}

    ax.axvspan(28.5, 29.5, alpha=0.15, color='red', label='DRAM cliff')

    # Annotate cliff ratios
    for key, offset in [('C', 0.8), ('F', 0.4)]:
        s = series.get(key, {})
        if 28 in s and 29 in s:
            ratio = s[29] / s[28]
            ax.annotate(f'{key}: {s[28]:.0f}s→{s[29]:.0f}s\n({ratio:.1f}×)',
                        xy=(29, s[29]),
                        xytext=(26.5, s[29] * (1.5 if key == 'C' else 0.6)),
                        arrowprops=dict(arrowstyle='->', color=COLORS[key], lw=1.5),
                        fontsize=8, color=COLORS[key])

    # J annotation — no cliff
    sj = series.get('J', {})
    if 28 in sj and 29 in sj:
        ratio = sj[29] / sj[28]
        ax.annotate(f'J (direct-index GPU): {ratio:.1f}× — no cliff',
                    xy=(29, sj[29]),
                    xytext=(26.5, sj[29] * 1.8),
                    arrowprops=dict(arrowstyle='->', color=COLORS['J'], lw=1.5),
                    fontsize=8, color=COLORS['J'])

    # State vector size labels
    for qq, sv in [(25, '268 MB'), (27, '1.1 GB'), (28, '2.1 GB'),
                   (29, '4.3 GB'), (30, '8.6 GB')]:
        ax.text(qq, 0.003, sv, fontsize=7, ha='center', color='grey')

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Wall-Clock Time (seconds, log scale)')
    ax.set_title('QFT DRAM Bandwidth Cliff — 28q→29q Phase Transition\n'
                 'direct-index GPU maintains smooth 2× scaling through 30q')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(24.5, 30.5)
    ax.set_xticks(range(25, 31))
    ax.set_yscale('log')
    fig.tight_layout()
    save(fig, "fig_qft3_cliff_zoom")


# ── Fig 4: direct-index vs Tensor (GPU and CPU) ──────────────────────
def fig_direct_index_vs_tensor(bk):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    pairs = [
        ('F', 'J', 'GPU: Tensor vs direct-index (Metal)', axes[0]),
        ('G', 'K', 'CPU: Tensor vs direct-index (MLX)',   axes[1]),
    ]

    for tensor_key, direct_index_key, title, ax in pairs:
        for key, style, lw in [(tensor_key, '-', 2), (direct_index_key, '-', 2.5)]:
            if key not in bk:
                continue
            q, t = _xy(bk[key])
            ax.semilogy(q, t, style, color=COLORS[key],
                        marker=MARKERS[key], label=LABELS[key], lw=lw)

        # Shade the cliff region
        ax.axvspan(28.5, 29.5, alpha=0.1, color='red')
        ax.text(29, ax.get_ylim()[1] if ax.get_ylim()[1] > 1 else 1000,
                'DRAM\ncliff', ha='center', fontsize=8, color='red')

        # Annotate speedup at 30q or last common point
        t_d  = {r['qubits']: r['time'] for r in bk.get(tensor_key, [])}
        a_d  = {r['qubits']: r['time'] for r in bk.get(direct_index_key,   [])}
        q_max = max(set(t_d) & set(a_d), default=None)
        if q_max and direct_index_key == 'J':
            ratio = t_d[q_max] / a_d[q_max]
            ax.annotate(f'{q_max}q: {ratio:.1f}× faster\n(direct-index)',
                        xy=(q_max, a_d[q_max]),
                        xytext=(q_max - 5, a_d[q_max] * 4),
                        arrowprops=dict(arrowstyle='->', color=COLORS[direct_index_key]),
                        fontsize=9, color=COLORS[direct_index_key])

        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Time (seconds, log scale)')
        ax.set_title(title)
        ax.legend(framealpha=0.9)
        ax.set_xlim(3, 31)

    fig.suptitle('direct-index Algorithm vs Tensordot: Same Hardware, Different Access Pattern\n'
                 'direct-index linear scan eliminates DRAM cliff on both GPU and CPU',
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_qft4_direct_index_vs_tensor")


# ── Fig 5: CPU Double Cliff ────────────────────────────────────
def fig_cpu_double_cliff(bk):
    fig, ax = plt.subplots(figsize=(11, 6))

    for key, ls in [('G', '-'), ('I', '--'), ('K', '-.')]:
        if key not in bk:
            continue
        q, t = _xy(bk[key])
        ax.semilogy(q, t, ls, color=COLORS[key],
                    marker=MARKERS[key], label=LABELS[key])

    # Cliff 1: L3 cache — 20→21q
    ax.axvspan(20.5, 21.5, alpha=0.15, color='#F39C12', label='L3 cache cliff (~20→21q)')
    ax.text(21, 0.0003, '21q\nL3 miss', ha='center', fontsize=8, color='#F39C12')

    # Cliff 2: DRAM — 27→28q (for tensor) or 28→29q
    ax.axvspan(27.5, 28.5, alpha=0.12, color='red', label='DRAM cliff (27→28q, tensor)')
    ax.text(28, 0.0003, '28q\nDRAM', ha='center', fontsize=8, color='red')

    # Annotate G's two cliff ratios
    g = {r['qubits']: r['time'] for r in bk.get('G', [])}
    if 20 in g and 21 in g:
        r1 = g[21] / g[20]
        ax.annotate(f'G: {r1:.1f}× at 21q', xy=(21, g[21]),
                    xytext=(23, g[21] * 0.3),
                    arrowprops=dict(arrowstyle='->', color=COLORS['G']),
                    fontsize=8, color=COLORS['G'])
    if 27 in g and 28 in g:
        r2 = g[28] / g[27]
        ax.annotate(f'G: {r2:.1f}× at 28q', xy=(28, g[28]),
                    xytext=(25.5, g[28] * 1.5),
                    arrowprops=dict(arrowstyle='->', color=COLORS['G']),
                    fontsize=8, color=COLORS['G'])

    # K has no cliff — annotate
    k = {r['qubits']: r['time'] for r in bk.get('K', [])}
    if 20 in k and 21 in k:
        r_k = k[21] / k[20]
        ax.text(21.5, k[21] * 0.5,
                f'K (direct-index): {r_k:.2f}× — no cliff',
                fontsize=8, color=COLORS['K'])

    ax.set_xlabel('Number of Qubits')
    ax.set_ylabel('Time (seconds, log scale)')
    ax.set_title('MLX CPU Backends: Two Cache Transitions\n'
                 'Tensor state representation triggers L3 and DRAM cliffs; direct-index avoids both')
    ax.legend(loc='upper left', framealpha=0.9)
    ax.set_xlim(3, 31)
    ax.set_xticks(range(3, 31, 2))
    fig.tight_layout()
    save(fig, "fig_qft5_cpu_double_cliff")


# ── Fig 6: 30q Summary Bar Chart ──────────────────────────────
def fig_30q_summary(bk):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Collect last-completed-qubit time for each backend
    summary = {}
    for key in 'ABCDEFGHIJK':
        rows = bk.get(key, [])
        if rows:
            last = rows[-1]
            summary[key] = (last['qubits'], last['time'])

    # Left: absolute time at max qubit reached
    ax = axes[0]
    keys   = list(summary.keys())
    qtimes = [summary[k][1] for k in keys]
    qlabels = [f"{LABELS[k].split(':')[0]}: {summary[k][0]}q" for k in keys]
    colors  = [COLORS[k] for k in keys]
    bars = ax.barh(qlabels, qtimes, color=colors, alpha=0.85, edgecolor='white')
    ax.axvline(600, color='grey', linestyle='--', lw=1.5, alpha=0.7,
               label='600s timeout')
    for bar, (key, (q, t)) in zip(bars, summary.items()):
        ax.text(min(t + 20, max(qtimes) * 0.98), bar.get_y() + bar.get_height() / 2,
                f'{t:.0f}s', va='center', fontsize=8)
    ax.set_xlabel('Wall-Clock Time (seconds)')
    ax.set_title('Time at Largest Completed Qubit Count')
    ax.legend()
    ax.set_xscale('log')

    # Right: effective bandwidth at 28q (last in-cache point for most backends)
    ax = axes[1]
    bw28 = {}
    for key in 'CDFGHJK':
        rows = bk.get(key, [])
        match = [r for r in rows if r['qubits'] == 28]
        if match:
            bw28[key] = match[0]['eff_gb_s']

    k28  = list(bw28.keys())
    bws  = [bw28[k] for k in k28]
    lbls = [LABELS[k].split(':')[0] for k in k28]
    clrs = [COLORS[k] for k in k28]
    bars = ax.barh(lbls, bws, color=clrs, alpha=0.85, edgecolor='white')
    for bar, (key, bw) in zip(bars, bw28.items()):
        ax.text(bw + 0.1, bar.get_y() + bar.get_height() / 2,
                f'{bw:.1f}', va='center', fontsize=8)
    ax.set_xlabel('Effective Bandwidth at 28q (GB/s)')
    ax.set_title('Effective Bandwidth at 28q\n(last fully in-DRAM point before cliff)')

    fig.suptitle('QFT 30-Qubit Performance Summary — Apple M4 Pro',
                 fontsize=13)
    fig.tight_layout()
    save(fig, "fig_qft6_30q_summary")


# ── GHZ data loader ───────────────────────────────────────────
GHZ_JSON = (Path(__file__).parent.parent / "results" /
            "exp_20260422_214851" / "inference" / "results.json")

GHZ_KEY_MAP = {
    'A_BruteForce_NumPy':    'A',
    'B_pykronecker':         'B',
    'C_JAX_tensordot':       'C',
    'D_direct-index_Reconstruction': 'D',
    'E_External_SSD':        'E',
    'F_MLX_GPU_Tensor':      'F',
    'G_MLX_CPU_Tensor':      'G',
    'H_MLX_GPU_Flat':        'H',
    'I_MLX_CPU_Flat':        'I',
    'J_MLX_GPU_direct-index':        'J',
    'K_MLX_CPU_direct-index':        'K',
}

def load_ghz(json_path=GHZ_JSON):
    """Return dict of letter → sorted list of {qubits, time_mean, time_std}."""
    raw = json.loads(Path(json_path).read_text())
    bk  = {}
    for r in raw:
        key = GHZ_KEY_MAP.get(r['algorithm'])
        if key is None:
            continue
        bk.setdefault(key, []).append({
            'qubits':    r['qubits'],
            'time_mean': r['time_mean'],
            'time_std':  r['time_std'],
        })
    for key in bk:
        bk[key].sort(key=lambda r: r['qubits'])
    return bk


# ── Fig 7: DRAM Cliff — GHZ vs QFT ───────────────────────────
def fig_cliff_ghz_vs_qft(qft_bk, ghz_bk):
    """Side-by-side cliff zoom (24–30q): same backends, same visual style.

    Shows how the 28→29q DRAM cliff appears in both circuits and whether
    direct-index GPU (J) escapes it in both cases.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    show     = ['C', 'F', 'J']
    datasets = [
        ('GHZ Circuit  (O(n) gates)',  ghz_bk,  'time_mean', axes[0]),
        ('QFT Circuit  (O(n²) gates)', qft_bk,  'time',      axes[1]),
    ]

    for title, bk, tkey, ax in datasets:
        series = {}
        for key in show:
            rows = [r for r in bk.get(key, []) if 24 <= r['qubits'] <= 30]
            if not rows:
                continue
            qs = [r['qubits'] for r in rows]
            ts = [r[tkey]     for r in rows]
            ax.semilogy(qs, ts, '-', color=COLORS[key],
                        marker=MARKERS[key], label=LABELS[key], lw=2)
            series[key] = {r['qubits']: r[tkey] for r in rows}

        ax.axvspan(28.5, 29.5, alpha=0.15, color='red')

        # Annotate cliff ratios at 28→29q
        for key in ['C', 'F', 'J']:
            s = series.get(key, {})
            if 28 in s and 29 in s:
                ratio = s[29] / s[28]
                ypos  = s[29]
                sign  = '' if ratio > 2.5 else ' (no cliff)'
                ax.annotate(f'{key}: {ratio:.1f}×{sign}',
                            xy=(29, ypos),
                            xytext=(27.2, ypos * (1.6 if key == 'C' else 0.55)),
                            arrowprops=dict(arrowstyle='->', color=COLORS[key], lw=1.5),
                            fontsize=8, color=COLORS[key])

        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Wall-Clock Time (seconds, log scale)')
        ax.set_title(title)
        ax.legend(loc='upper left', framealpha=0.9)
        ax.set_xlim(24.5, 30.5)
        ax.set_xticks(range(25, 31))

        # State size labels at bottom
        for qq, sv in [(25, '268 MB'), (27, '1.1 GB'), (28, '2.1 GB'),
                       (29, '4.3 GB'), (30, '8.6 GB')]:
            ax.text(qq, ax.get_ylim()[0] * 1.3, sv,
                    fontsize=7, ha='center', color='grey')

    fig.suptitle('DRAM Bandwidth Cliff: GHZ vs QFT Circuits\n'
                 'Both circuits hit the 28→29q cliff; direct-index GPU avoids it in both',
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_qft7_cliff_ghz_vs_qft")


# ── Fig 8: GHZ vs QFT Performance Comparison ─────────────────
def fig_ghz_vs_qft(qft_bk, ghz_bk):
    """Two panels (GPU / CPU): GHZ dashed vs QFT solid for matched backends.

    Also plots the expected QFT/GHZ ratio from gate-count theory:
        GHZ gates ≈ n   (1 H + n-1 CNOTs)
        QFT gates ≈ n²/2  (n H + n(n-1)/2 CP + n/2 SWAP)
        Expected ratio ≈ n/2
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    panels = [
        ('GPU Backends', ['F', 'J'], axes[0]),
        ('CPU Backends', ['C', 'D'], axes[1]),
    ]

    for panel_title, keys, ax in panels:
        for key in keys:
            ghz_rows = ghz_bk.get(key, [])
            qft_rows = qft_bk.get(key, [])
            if ghz_rows:
                qs = [r['qubits']    for r in ghz_rows]
                ts = [r['time_mean'] for r in ghz_rows]
                ax.semilogy(qs, ts, '--', color=COLORS[key], lw=1.8,
                            marker=MARKERS[key], markevery=4, alpha=0.7,
                            label=f'{LABELS[key]} — GHZ')
            if qft_rows:
                qs = [r['qubits'] for r in qft_rows]
                ts = [r['time']   for r in qft_rows]
                ax.semilogy(qs, ts, '-', color=COLORS[key], lw=2.5,
                            marker=MARKERS[key], markevery=4,
                            label=f'{LABELS[key]} — QFT')

        ax.axvspan(28.5, 29.5, alpha=0.10, color='red', label='DRAM cliff')
        ax.set_xlabel('Number of Qubits')
        ax.set_ylabel('Wall-Clock Time (seconds, log scale)')
        ax.set_title(panel_title)
        ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
        ax.set_xlim(3, 31)

    fig.suptitle('GHZ vs QFT: Wall-Clock Time per Backend\n'
                 'Dashed = GHZ  |  Solid = QFT  |  QFT is slower due to O(n²) gate count',
                 fontsize=12)
    fig.tight_layout()
    save(fig, "fig_qft8_ghz_vs_qft_time")

    # ── Companion: QFT/GHZ ratio vs qubits ──────────────────
    fig2, ax2 = plt.subplots(figsize=(10, 6))

    # Only plot from 21q onward: below that, C (JAX) ratio is ~1000× because
    # GHZ was benchmarked over 7 trials (JIT amortised) while QFT is a single
    # run (JIT overhead included every time), making small-q ratios meaningless.
    show_ratio = ['C', 'F', 'J', 'K']
    q_theory   = np.arange(21, 31)
    ax2.plot(q_theory, q_theory / 2, 'k--', lw=1.5, alpha=0.5,
             label='Expected ratio ≈ n/2  (gate-count model)')

    for key in show_ratio:
        ghz_map = {r['qubits']: r['time_mean'] for r in ghz_bk.get(key, [])}
        qft_map = {r['qubits']: r['time']      for r in qft_bk.get(key, [])}
        common  = sorted(q for q in set(ghz_map) & set(qft_map) if q >= 21)
        if not common:
            continue
        ratios = [qft_map[q] / ghz_map[q] for q in common]
        ax2.plot(common, ratios, '-', color=COLORS[key],
                 marker=MARKERS[key], label=LABELS[key])

    ax2.axvspan(28.5, 29.5, alpha=0.10, color='red', label='DRAM cliff')
    ax2.text(22.5, 2,
             'Note: ratio shown from 21q only.\n'
             'Below 21q, JAX (C) ratio is >1000× — an artifact:\n'
             'GHZ used 7-trial mean (JIT amortised);\n'
             'QFT is a single run (JIT overhead included each time).',
             fontsize=8, style='italic',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.85))
    ax2.set_xlabel('Number of Qubits')
    ax2.set_ylabel('QFT Time / GHZ Time  (ratio)')
    ax2.set_title('How Much Harder is QFT than GHZ?\n'
                  'Bandwidth-bound regime (21–30q): ratio converges to ~n/2 for all backends')
    ax2.legend(loc='upper left', framealpha=0.9)
    ax2.set_xlim(20, 31)
    ax2.set_xticks(range(21, 31))
    fig2.tight_layout()
    save(fig2, "fig_qft9_ghz_vs_qft_ratio")


# ── Main ──────────────────────────────────────────────────────
def main():
    print("\n── QFT Benchmark Figures ───────────────────────────────")
    bk = parse_log()
    print(f"  Parsed {len(bk)} backends: {', '.join(sorted(bk))}")
    for key in sorted(bk):
        rows = bk[key]
        print(f"    {key}: {rows[0]['qubits']}q–{rows[-1]['qubits']}q "
              f"({len(rows)} points)")

    ghz_bk = load_ghz()
    print(f"  Loaded GHZ: {len(ghz_bk)} backends")

    print("\nGenerating figures:")
    fig_all_backends(bk)
    fig_eff_bandwidth(bk)
    fig_cliff_zoom(bk)
    fig_direct_index_vs_tensor(bk)
    fig_cpu_double_cliff(bk)
    fig_30q_summary(bk)
    fig_cliff_ghz_vs_qft(bk, ghz_bk)
    fig_ghz_vs_qft(bk, ghz_bk)

    print(f"\n  All figures saved to {OUTDIR}")
    print("  Format: PNG, 300 DPI (paper-ready)")


if __name__ == "__main__":
    main()
