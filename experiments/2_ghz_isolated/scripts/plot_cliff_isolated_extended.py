"""
plot_cliff_isolated_5.py

Four figures from the thermally-isolated GHZ cliff benchmark (Exp 5).
Data source: cliff_isolated_ghz_20260506_131012.csv
Backends:    F, G (tensordot), H, I (flat-index), J, K (direct-index)

Fig 1  — Wall-clock time vs qubits (log y, error bars, all 6 backends)
Fig 2  — Effective bandwidth vs qubits (all 6 backends)
Fig 3  — Qubit-step scaling ratios (28/27, 29/28, 30/29 per backend)
Fig 4  — GPU speedup per algorithm class vs STREAM prediction

Run: python3 scripts/plot_cliff_isolated_5.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "data"
OUT_DIR  = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

CSV_PATH = DATA_DIR / "cliff_isolated_ghz_20260506_131012.csv"

# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv(path):
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            bk = row['backend']
            entry = {
                'qubits': int(row['qubits']),
                'mean':   float(row['mean_s']),
                'std':    float(row['std_s']),
                'eff_bw': float(row['eff_bw_gb_s']),
                'trials': [float(row[f't{i}']) for i in range(1, 6) if row.get(f't{i}')],
            }
            data.setdefault(bk, []).append(entry)
    for bk in data:
        data[bk].sort(key=lambda r: r['qubits'])
    return data

DATA   = load_csv(CSV_PATH)
QUBITS = [27, 28, 29, 30]

# ── Style ─────────────────────────────────────────────────────────────────────
# GPU: solid line  |  CPU: dashed line
# Tensordot: orange family  |  Flat-index: teal family  |  Direct-index: blue/purple
STYLE = {
    'F': dict(color='#E65100', marker='s', ls='-',  lw=2.0, label='F: MLX GPU tensordot',    zorder=5),
    'G': dict(color='#FF8F00', marker='s', ls='--', lw=1.8, label='G: MLX CPU tensordot',    zorder=4),
    'H': dict(color='#00695C', marker='^', ls='-',  lw=2.0, label='H: MLX GPU flat-index',   zorder=5),
    'I': dict(color='#26A69A', marker='^', ls='--', lw=1.8, label='I: MLX CPU flat-index',   zorder=4),
    'J': dict(color='#1565C0', marker='o', ls='-',  lw=2.2, label='J: MLX GPU direct-index', zorder=6),
    'K': dict(color='#7B1FA2', marker='D', ls='--', lw=1.8, label='K: MLX CPU direct-index', zorder=3),
}
MS = 7

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'figure.dpi':     150,
})

XTICK_LABELS = [f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS]

# ── Fig 1: Wall-clock time (log y) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for bk in ['F', 'G', 'H', 'I', 'J', 'K']:
    rows = DATA[bk]
    q = [r['qubits'] for r in rows]
    m = [r['mean']   for r in rows]
    e = [r['std']    for r in rows]
    st = STYLE[bk]
    ax.errorbar(q, m, yerr=e, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=MS, capsize=4, capthick=1.2,
                label=st['label'], zorder=st['zorder'])

# Cliff annotations at 28→29q
cliff_annots = {
    'F': (3.16, +0.40), 'H': (4.03, +0.55),
    'J': (2.09, -0.38), 'K': (2.07, -0.55),
}
for bk, (ratio, dy_log) in cliff_annots.items():
    r28 = next(r for r in DATA[bk] if r['qubits'] == 28)
    r29 = next(r for r in DATA[bk] if r['qubits'] == 29)
    y_mid = np.sqrt(r28['mean'] * r29['mean'])
    ax.annotate(f'{ratio:.2f}×', xy=(28.5, y_mid),
                xytext=(28.5, y_mid * (10**dy_log)),
                fontsize=8, color=STYLE[bk]['color'], ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=STYLE[bk]['color'],
                                lw=0.8, connectionstyle='arc3,rad=0.0'))

ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5, zorder=1)
ax.text(28.52, 0.9, 'DRAM cliff\n28→29q', fontsize=8, color='red',
        alpha=0.7, va='bottom', transform=ax.get_xaxis_transform())

ax.set_yscale('log')
ax.set_xticks(QUBITS)
ax.set_xticklabels(XTICK_LABELS)
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('GHZ Circuit — Wall-clock Time, Thermally Isolated (N=5)\n'
             'Solid = GPU  |  Dashed = CPU  |  Cliff at 28→29q for tensordot & flat-index')
ax.legend(loc='upper left', framealpha=0.9, ncol=2)
ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.7, 30.3)

fig.tight_layout()
out1 = OUT_DIR / "fig1_wall_time.png"
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out1}")


# ── Fig 2: Effective bandwidth ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))

for bk in ['F', 'G', 'H', 'I', 'J', 'K']:
    rows = DATA[bk]
    q  = [r['qubits'] for r in rows]
    bw = [r['eff_bw'] for r in rows]
    st = STYLE[bk]
    ax.plot(q, bw, color=st['color'], marker=st['marker'],
            ls=st['ls'], lw=st['lw'], ms=MS, label=st['label'], zorder=st['zorder'])
    for qv, bval in zip(q, bw):
        offset = +8 if bk in ('J',) else -13
        ax.annotate(f'{bval:.1f}', xy=(qv, bval),
                    xytext=(0, offset), textcoords='offset points',
                    fontsize=7, color=st['color'], ha='center', alpha=0.85)

# STREAM reference lines
ax.axhline(y=221.9, color='#1565C0', lw=1.0, ls=':', alpha=0.5)
ax.axhline(y=119.9, color='#7B1FA2', lw=1.0, ls=':', alpha=0.5)
ax.text(0.99, 221.9 / 65, 'GPU STREAM 221.9', fontsize=7.5, color='#1565C0',
        alpha=0.7, va='bottom', ha='right', transform=ax.get_yaxis_transform())
ax.text(0.99, 119.9 / 65, 'CPU STREAM 119.9', fontsize=7.5, color='#7B1FA2',
        alpha=0.7, va='bottom', ha='right', transform=ax.get_yaxis_transform())

ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5)
ax.text(0.42, 0.97, 'DRAM cliff', transform=ax.transAxes,
        fontsize=8, color='red', alpha=0.7, va='top', ha='center')

ax.set_xticks(QUBITS)
ax.set_xticklabels(XTICK_LABELS)
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Effective memory bandwidth (GB/s)')
ax.set_title('GHZ Circuit — Effective Bandwidth, Thermally Isolated\n'
             'Direct-index (J) sustains ~47 GB/s throughout; tensordot drops at cliff')
ax.legend(loc='upper right', framealpha=0.9, ncol=2)
ax.grid(True, alpha=0.25)
ax.set_xlim(26.7, 30.3)
ax.set_ylim(0, 65)

fig.tight_layout()
out2 = OUT_DIR / "fig2_eff_bandwidth.png"
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out2}")


# ── Fig 3: Qubit-step scaling ratios ─────────────────────────────────────────
steps  = ['27→28', '28→29', '29→30']
x      = np.arange(len(steps))
width  = 0.12
offsets = np.array([-2.5, -1.5, -0.5, 0.5, 1.5, 2.5]) * width
order  = ['F', 'G', 'H', 'I', 'J', 'K']

fig, ax = plt.subplots(figsize=(9, 5.5))

for idx, bk in enumerate(order):
    rows   = DATA[bk]
    means  = [r['mean'] for r in rows]
    ratios = [means[i+1] / means[i] for i in range(3)]
    bars   = ax.bar(x + offsets[idx], ratios, width,
                    color=STYLE[bk]['color'], label=STYLE[bk]['label'],
                    alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=7.5,
                color=STYLE[bk]['color'], fontweight='bold', rotation=90)

ax.axhline(y=2.0, color='gray', lw=1.2, ls='--', alpha=0.6, label='2× ideal doubling')
ax.axhline(y=3.0, color='red',  lw=0.8, ls=':',  alpha=0.4)
ax.text(2.48, 3.05, 'cliff threshold (3×)', fontsize=8, color='red', alpha=0.6)

ax.axvspan(0.5, 1.5, alpha=0.06, color='red', zorder=0)
ax.text(1.0, 5.1, 'DRAM\ncliff', ha='center', fontsize=8.5, color='red', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=11)
ax.set_ylabel('Scaling ratio  t(q) / t(q−1)')
ax.set_title('GHZ — Qubit-step Scaling Ratios, Thermally Isolated\n'
             'F cliff 3.16×, H cliff 4.03×, I early cliff 3.45× at 27→28q; J/K flat ~2×')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax.set_ylim(0, 5.8)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out3 = OUT_DIR / "fig3_step_ratios.png"
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out3}")


# ── Fig 4: GPU speedup vs STREAM prediction ───────────────────────────────────
# Pairs: (CPU backend, GPU backend, algorithm label, GPU color)
pairs = [
    ('G', 'F', 'Tensordot\n(G vs F)',    '#FF8F00'),
    ('I', 'H', 'Flat-index\n(I vs H)',   '#26A69A'),
    ('K', 'J', 'Direct-index\n(K vs J)', '#1565C0'),
]
STREAM_PRED = 1.85
q_labels = ['27q', '28q', '29q', '30q']
x = np.arange(len(QUBITS))
width = 0.22
offsets = np.array([-1, 0, 1]) * width

fig, ax = plt.subplots(figsize=(9, 5.5))

for idx, (cpu_bk, gpu_bk, lbl, col) in enumerate(pairs):
    cpu_means = [r['mean'] for r in DATA[cpu_bk]]
    gpu_means = [r['mean'] for r in DATA[gpu_bk]]
    speedups  = [c / g for c, g in zip(cpu_means, gpu_means)]
    bars = ax.bar(x + offsets[idx], speedups, width,
                  color=col, label=lbl, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.08,
                f'{val:.2f}×', ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')

ax.axhline(y=STREAM_PRED, color='black', lw=1.5, ls='--', alpha=0.7,
           label=f'STREAM prediction: {STREAM_PRED}×\n(MLX CPU 119.9 / MLX GPU 221.9 GB/s)')
ax.text(3.55, STREAM_PRED + 0.08, f'{STREAM_PRED}×\nSTREAM', fontsize=8.5,
        color='black', alpha=0.7, ha='left', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(q_labels, fontsize=11)
ax.set_ylabel('GPU Speedup  (CPU time / GPU time)')
ax.set_title('GHZ — GPU Speedup vs STREAM-predicted 1.85×\n'
             'All algorithm classes exceed STREAM prediction; direct-index ~10× throughout')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9.5)
ax.set_ylim(0, 12.5)
ax.grid(True, axis='y', alpha=0.25)

# Annotate the 28q spike for flat-index
i_means = [r['mean'] for r in DATA['I']]
h_means = [r['mean'] for r in DATA['H']]
spike = i_means[1] / h_means[1]
ax.annotate('Mismatched\ncliff spike\n(I at 27→28q,\nH at 28→29q)',
            xy=(x[1] + offsets[1], spike),
            xytext=(x[1] + offsets[1] - 0.6, spike + 1.5),
            fontsize=8, color='#26A69A',
            arrowprops=dict(arrowstyle='->', color='#26A69A', lw=0.9))

fig.tight_layout()
out4 = OUT_DIR / "fig4_gpu_speedup.png"
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out4}")

print(f"\nAll 4 figures saved to: {OUT_DIR}")
