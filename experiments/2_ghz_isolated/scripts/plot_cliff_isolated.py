"""
plot_cliff_isolated.py

Four figures from the thermally-isolated GHZ cliff benchmark.
Data source: cliff_isolated_ghz_20260428_094411.csv

Fig 1  — Wall-clock time vs qubits (log y, error bars, all 4 backends)
Fig 2  — Effective bandwidth vs qubits (all 4 backends)
Fig 3  — Qubit-step scaling ratios  (28/27, 29/28, 30/29 per backend)
Fig 4  — JAX original (thermal) vs thermally-isolated comparison

Run: python3 plot_cliff_isolated.py
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

# ── Data ──────────────────────────────────────────────────────────────────────
CSV_PATH = Path(__file__).parent.parent / "cliff_isolated_ghz_20260428_094411.csv"
OUT_DIR  = Path(__file__).parent.parent / "results" / "figures_isolated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Original thermally-contaminated data (quantum_benchmark.py, N=7, Table 4)
# Only C (JAX CPU) is documented with anomaly; paper cites 28q/29q/30q
ORIG_C = {
    'qubits': [28, 29, 30],
    'mean':   [9.88, 52.17, 38.33],
    'std':    [0.04,  0.16,  0.03],
}

def load_csv(path):
    """Returns dict: backend_key → {qubits, mean, std, eff_bw, trials}."""
    data = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            bk  = row['backend']
            n   = int(row['qubits'])
            ts  = [float(row[f't{i}']) for i in range(1, 6) if row.get(f't{i}')]
            entry = {
                'qubits':  n,
                'mean':    float(row['mean_s']),
                'std':     float(row['std_s']),
                'eff_bw':  float(row['eff_bw_gb_s']),
                'trials':  ts,
            }
            data.setdefault(bk, []).append(entry)
    # Sort by qubit count
    for bk in data:
        data[bk].sort(key=lambda r: r['qubits'])
    return data


DATA = load_csv(CSV_PATH)
QUBITS = [27, 28, 29, 30]

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    'C': dict(color='#2196F3', marker='o', ls='-',  lw=2.0, label='C: JAX CPU tensordot',  zorder=4),
    'F': dict(color='#FF9800', marker='s', ls='-',  lw=2.0, label='F: MLX GPU tensor',      zorder=3),
    'J': dict(color='#4CAF50', marker='^', ls='-',  lw=2.5, label='J: MLX GPU direct-index',        zorder=5),
    'K': dict(color='#9C27B0', marker='D', ls='--', lw=1.8, label='K: MLX CPU direct-index',        zorder=2),
}
MS = 7   # marker size

plt.rcParams.update({
    'font.family':   'sans-serif',
    'font.size':     11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi':    150,
})

# ── Fig 1: Wall-clock time (log y) ────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

for bk, rows in DATA.items():
    q  = [r['qubits'] for r in rows]
    m  = [r['mean']   for r in rows]
    e  = [r['std']    for r in rows]
    st = STYLE[bk]
    ax.errorbar(q, m, yerr=e, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=MS, capsize=4, capthick=1.2,
                label=st['label'], zorder=st['zorder'])

# Annotate cliff ratios at 28→29q
cliff_annots = {'C': (4.46, 'up'), 'F': (3.15, 'up'), 'J': (2.15, 'down'), 'K': (2.06, 'down')}
for bk, (ratio, side) in cliff_annots.items():
    r28 = next(r for r in DATA[bk] if r['qubits'] == 28)
    r29 = next(r for r in DATA[bk] if r['qubits'] == 29)
    y_mid = np.sqrt(r28['mean'] * r29['mean'])
    dy = 0.45 if side == 'up' else -0.45
    ax.annotate(f'{ratio:.2f}×', xy=(28.5, y_mid),
                xytext=(28.5, y_mid * (10**dy)),
                fontsize=8.5, color=STYLE[bk]['color'],
                ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=STYLE[bk]['color'],
                                lw=0.8, connectionstyle='arc3,rad=0.0'))

# Cliff shading
ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5, zorder=1)
ax.text(28.52, ax.get_ylim()[0] if False else 0.9, 'DRAM cliff\n28→29q',
        fontsize=8, color='red', alpha=0.7, va='bottom', transform=ax.get_xaxis_transform())

ax.set_yscale('log')
ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('GHZ Circuit: Wall-clock Time — Thermally Isolated, N=5 trials\n'
             'Cliff exists (4.46× JAX, 3.15× MLX GPU tensor); no 30q < 29q anomaly')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.7, 30.3)

# Monotone arrow annotation for 30q
for bk in ['C', 'F', 'J', 'K']:
    r29 = next(r for r in DATA[bk] if r['qubits'] == 29)
    r30 = next(r for r in DATA[bk] if r['qubits'] == 30)
    if r30['mean'] > r29['mean'] * 1.01:
        ax.annotate('', xy=(30, r30['mean']), xytext=(29, r29['mean']),
                    arrowprops=dict(arrowstyle='->', color=STYLE[bk]['color'],
                                    lw=1.2, alpha=0.4))

fig.tight_layout()
out1 = OUT_DIR / "cliff_isolated_fig1_wall_time.png"
fig.savefig(out1, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out1}")


# ── Fig 2: Effective bandwidth ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

for bk, rows in DATA.items():
    q  = [r['qubits'] for r in rows]
    bw = [r['eff_bw'] for r in rows]
    st = STYLE[bk]
    ax.plot(q, bw, color=st['color'], marker=st['marker'],
            ls=st['ls'], lw=st['lw'], ms=MS, label=st['label'], zorder=st['zorder'])
    # Mark min/max to show variability
    for qv, bval in zip(q, bw):
        ax.annotate(f'{bval:.1f}', xy=(qv, bval),
                    xytext=(0, 7 if bk in ('J',) else -12),
                    textcoords='offset points',
                    fontsize=7.5, color=st['color'], ha='center', alpha=0.85)

ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5)
ax.text(0.42, 0.97, 'DRAM cliff', transform=ax.transAxes,
        fontsize=8, color='red', alpha=0.7, va='top', ha='center')

ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Effective memory bandwidth (GB/s)')
ax.set_title('GHZ Circuit: Effective Bandwidth — Thermally Isolated\n'
             'direct-index maintains ~47 GB/s (GPU) through cliff; tensordot drops 28→13 GB/s')
ax.legend(loc='center right', framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim(26.7, 30.3)
ax.set_ylim(0, 60)

# Flat-line annotation for J
j_bw = [r['eff_bw'] for r in DATA['J']]
ax.annotate('direct-index: cliff-resistant\n~47 GB/s throughout',
            xy=(29, j_bw[2]), xytext=(27.5, 54),
            fontsize=8.5, color=STYLE['J']['color'],
            arrowprops=dict(arrowstyle='->', color=STYLE['J']['color'], lw=0.9))

fig.tight_layout()
out2 = OUT_DIR / "cliff_isolated_fig2_eff_bandwidth.png"
fig.savefig(out2, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out2}")


# ── Fig 3: Qubit-step scaling ratios ─────────────────────────────────────────
steps  = ['27→28', '28→29', '29→30']
x      = np.arange(len(steps))
width  = 0.18
offset = np.array([-1.5, -0.5, 0.5, 1.5]) * width

fig, ax = plt.subplots(figsize=(8, 5))

for idx, (bk, rows) in enumerate(DATA.items()):
    means = [r['mean'] for r in rows]
    ratios = [means[i+1] / means[i] for i in range(len(means)-1)]
    bars = ax.bar(x + offset[idx], ratios, width,
                  color=STYLE[bk]['color'], label=STYLE[bk]['label'],
                  alpha=0.85, edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, ratios):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f'{val:.2f}×', ha='center', va='bottom', fontsize=8.5,
                color=STYLE[bk]['color'], fontweight='bold')

ax.axhline(y=2.0, color='gray', lw=1.2, ls='--', alpha=0.6, label='2× (ideal doubling)')
ax.axhline(y=4.0, color='red',  lw=1.0, ls=':',  alpha=0.4)
ax.text(2.45, 4.05, 'cliff threshold', fontsize=8, color='red', alpha=0.6)

# Shade the cliff step
ax.axvspan(0.5, 1.5, alpha=0.06, color='red', zorder=0)
ax.text(1.0, ax.get_ylim()[1]*0.92 if False else 4.7, 'DRAM\ncliff',
        ha='center', fontsize=8.5, color='red', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=11)
ax.set_ylabel('Scaling ratio (next_q / prev_q)')
ax.set_title('GHZ: Qubit-step Scaling Ratios — Thermally Isolated\n'
             'Cliff clearly at 28→29q; 30q step returns to ~2× (no anomaly)')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9.5)
ax.set_ylim(0, 5.5)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out3 = OUT_DIR / "cliff_isolated_fig3_step_ratios.png"
fig.savefig(out3, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out3}")


# ── Fig 4: JAX original (thermal artifact) vs thermally isolated ──────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

c_rows   = DATA['C']
c_qubits = [r['qubits'] for r in c_rows]
c_mean   = [r['mean']   for r in c_rows]
c_std    = [r['std']    for r in c_rows]

ax.errorbar(c_qubits, c_mean, yerr=c_std,
            color='#2196F3', marker='o', ls='-', lw=2.2, ms=8,
            capsize=4, capthick=1.2,
            label='New: thermally isolated, N=5 (this run)', zorder=4)

ax.errorbar(ORIG_C['qubits'], ORIG_C['mean'], yerr=ORIG_C['std'],
            color='#F44336', marker='X', ls='--', lw=1.8, ms=9,
            capsize=4, capthick=1.2, alpha=0.85,
            label='Original: thermal artifact, N=7 (Table 4)', zorder=3)

# Annotate the key differences
pairs = [(28, 9.88, 4.222), (29, 52.17, 18.846), (30, 38.33, 39.122)]
for q, orig, new in pairs:
    if orig > new:
        inflation = orig / new
        ax.annotate(f'{inflation:.1f}× inflation',
                    xy=(q, new), xytext=(q - 0.6, (orig + new) / 2),
                    fontsize=8.5, color='#F44336',
                    arrowprops=dict(arrowstyle='<->', color='#F44336', lw=1.0),
                    ha='right')
    else:
        ax.annotate(f'≈same\n({abs(orig-new)/new*100:.0f}% diff)',
                    xy=(q, new), xytext=(q + 0.2, new * 1.15),
                    fontsize=8, color='gray', ha='left')

# Anomaly annotation
ax.annotate('', xy=(30, 38.33), xytext=(29, 52.17),
            arrowprops=dict(arrowstyle='->', color='#F44336',
                            lw=1.5, connectionstyle='arc3,rad=-0.3'))
ax.text(29.6, 43, 'Thermal anomaly\n30q < 29q\n(now resolved)', fontsize=8.5,
        color='#F44336', ha='center', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FFEBEE', alpha=0.8))

ax.set_yscale('log')
ax.set_xticks([27, 28, 29, 30])
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in [27, 28, 29, 30]])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('JAX CPU tensordot: Thermal Artifact Exposed\n'
             'Original data inflated 28q by 2.3×, 29q by 2.8×; 30q anomaly disappears')
ax.legend(framealpha=0.9)
ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.6, 30.4)

# Corrected cliff ratio annotation
ax.text(0.97, 0.05,
        'Corrected cliff ratio: 4.46× (was 5.28×)\n'
        'True 28→29q DRAM wall — thermally clean',
        transform=ax.transAxes, fontsize=8.5, ha='right', va='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#E3F2FD', alpha=0.9))

fig.tight_layout()
out4 = OUT_DIR / "cliff_isolated_fig4_thermal_comparison.png"
fig.savefig(out4, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out4}")

print(f"\nAll figures saved to: {OUT_DIR}")
