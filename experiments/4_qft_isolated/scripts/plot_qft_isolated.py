"""
plot_qft_isolated.py

Four figures from the thermally-isolated QFT cross-validation benchmark.
Data sources (results/qft_isolated_final/):
  qft_C_jax.log                       — C (JAX CPU tensordot)
  qft_J_mlx_gpu_direct_index.log              — J (MLX GPU direct-index)
  qft_FK_mlx_gpu_tensor_cpu_direct_index.log  — F (MLX GPU tensor) + K (MLX CPU direct-index)

Fig 1 — Wall-clock time vs qubits (log y, error bars, all 4 backends)
Fig 2 — Effective bandwidth vs qubits
Fig 3 — Qubit-step scaling ratios (bar chart)
Fig 4 — Cross-circuit comparison: QFT cliff vs GHZ cliff (4 backends)

Run: python3 plot_qft_isolated.py
"""

import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).parent.parent / "results" / "qft_isolated_final"
OUT_DIR  = Path(__file__).parent.parent / "results" / "figures_qft_isolated"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Parser ────────────────────────────────────────────────────────────────────
ROW_RE = re.compile(
    r'^\s{2}(\d{2})\s+[\d.]+\s+'          # q, warmup
    r'([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+'  # t1, t2, t3
    r'([\d.]+)\s+([\d.]+)\s+'             # mean, std
    r'[\d.]+%\s+'                          # cov
    r'[\d.×—]+\s+'                         # ratio
    r'([\d.]+)'                            # eff_bw
)

BK_HDR = re.compile(r'QFT \| (\w+):')

def parse_log(path):
    """Returns dict: backend_key → list of {qubits, mean, std, eff_bw, trials}."""
    data = {}
    current = None
    for line in Path(path).read_text().splitlines():
        m = BK_HDR.search(line)
        if m:
            current = m.group(1)
            data[current] = []
            continue
        if current is None:
            continue
        m = ROW_RE.match(line)
        if m:
            q, t1, t2, t3, mean, std, bw = m.groups()
            data[current].append({
                'qubits': int(q),
                'mean':   float(mean),
                'std':    float(std),
                'eff_bw': float(bw),
                'trials': [float(t1), float(t2), float(t3)],
            })
    return data


def load_all():
    out = {}
    for fname, expected_keys in [
        ('qft_C_jax.log',                      ['C']),
        ('qft_J_mlx_gpu_direct_index.log',             ['J']),
        ('qft_FK_mlx_gpu_tensor_cpu_direct_index.log', ['F', 'K']),
    ]:
        parsed = parse_log(DATA_DIR / fname)
        for k in expected_keys:
            if k in parsed:
                out[k] = parsed[k]
    return out


DATA   = load_all()
QUBITS = [27, 28, 29, 30]

# GHZ isolated cliff ratios (from cliff_isolated_ghz_20260428_094411.log)
GHZ_CLIFF = {'C': 4.46, 'F': 3.15, 'J': 2.15, 'K': 2.06}

# ── Style ─────────────────────────────────────────────────────────────────────
STYLE = {
    'C': dict(color='#2196F3', marker='o', ls='-',  lw=2.0, label='C: JAX CPU tensordot',  zorder=4),
    'F': dict(color='#FF9800', marker='s', ls='-',  lw=2.0, label='F: MLX GPU tensor',      zorder=3),
    'J': dict(color='#4CAF50', marker='^', ls='-',  lw=2.5, label='J: MLX GPU direct-index',        zorder=5),
    'K': dict(color='#9C27B0', marker='D', ls='--', lw=1.8, label='K: MLX CPU direct-index',        zorder=2),
}
MS = 7

plt.rcParams.update({
    'font.family': 'sans-serif', 'font.size': 11,
    'axes.titlesize': 13, 'axes.labelsize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
})

# ── Fig 1: Wall-clock time ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

for bk, rows in DATA.items():
    q = [r['qubits'] for r in rows]
    m = [r['mean']   for r in rows]
    e = [r['std']    for r in rows]
    st = STYLE[bk]
    ax.errorbar(q, m, yerr=e, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=MS, capsize=4, capthick=1.2,
                label=st['label'], zorder=st['zorder'])

# Cliff ratio annotations at 28→29q
annots = {'C': (4.33, 1.0), 'F': (3.84, 0.6), 'J': (2.12, -0.5), 'K': (2.12, -1.0)}
for bk, (ratio, dy_exp) in annots.items():
    r28 = next(r for r in DATA[bk] if r['qubits'] == 28)
    r29 = next(r for r in DATA[bk] if r['qubits'] == 29)
    y_mid = np.sqrt(r28['mean'] * r29['mean'])
    ax.annotate(f'{ratio:.2f}×', xy=(28.5, y_mid),
                xytext=(28.5, y_mid * (10**dy_exp)),
                fontsize=8.5, color=STYLE[bk]['color'], ha='center', va='center',
                arrowprops=dict(arrowstyle='->', color=STYLE[bk]['color'], lw=0.8))

ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5)
ax.text(0.42, 0.97, 'DRAM cliff', transform=ax.transAxes,
        fontsize=8, color='red', alpha=0.7, va='top', ha='center')

ax.set_yscale('log')
ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('QFT Circuit: Wall-clock Time — Thermally Isolated, N=3 trials\n'
             'Cliff at 28→29q (4.33× JAX, 3.84× MLX GPU tensor); direct-index ~2.1× throughout')
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.7, 30.3)

fig.tight_layout()
out = OUT_DIR / "qft_isolated_fig1_wall_time.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")


# ── Fig 2: Effective bandwidth ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

for bk, rows in DATA.items():
    q  = [r['qubits'] for r in rows]
    bw = [r['eff_bw'] for r in rows]
    st = STYLE[bk]
    ax.plot(q, bw, color=st['color'], marker=st['marker'],
            ls=st['ls'], lw=st['lw'], ms=MS, label=st['label'], zorder=st['zorder'])
    for qv, bval in zip(q, bw):
        offset = 7 if bk == 'J' else -12
        ax.annotate(f'{bval:.1f}', xy=(qv, bval),
                    xytext=(0, offset), textcoords='offset points',
                    fontsize=7.5, color=st['color'], ha='center', alpha=0.85)

ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5)
ax.text(0.42, 0.97, 'DRAM cliff', transform=ax.transAxes,
        fontsize=8, color='red', alpha=0.7, va='top', ha='center')

ax.annotate('direct-index: cliff-resistant\n~13 GB/s throughout',
            xy=(29, next(r['eff_bw'] for r in DATA['J'] if r['qubits'] == 29)),
            xytext=(27.5, 16),
            fontsize=8.5, color=STYLE['J']['color'],
            arrowprops=dict(arrowstyle='->', color=STYLE['J']['color'], lw=0.9))

ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Effective memory bandwidth (GB/s)')
ax.set_title('QFT Circuit: Effective Bandwidth — Thermally Isolated\n'
             'direct-index flat ~13 GB/s (GPU) through cliff; tensordot drops 9.8→5.4 GB/s')
ax.legend(loc='center right', framealpha=0.9)
ax.grid(True, alpha=0.25)
ax.set_xlim(26.7, 30.3)
ax.set_ylim(0, 20)

fig.tight_layout()
out = OUT_DIR / "qft_isolated_fig2_eff_bandwidth.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")


# ── Fig 3: Qubit-step scaling ratios ─────────────────────────────────────────
steps  = ['27→28', '28→29', '29→30']
x      = np.arange(len(steps))
width  = 0.18
offset = np.array([-1.5, -0.5, 0.5, 1.5]) * width

fig, ax = plt.subplots(figsize=(8, 5))

for idx, (bk, rows) in enumerate(DATA.items()):
    means  = [r['mean'] for r in rows]
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
ax.axvspan(0.5, 1.5, alpha=0.06, color='red', zorder=0)
ax.text(1.0, 4.6, 'DRAM\ncliff', ha='center', fontsize=8.5, color='red', alpha=0.7)

ax.set_xticks(x)
ax.set_xticklabels(steps, fontsize=11)
ax.set_ylabel('Scaling ratio (next_q / prev_q)')
ax.set_title('QFT: Qubit-step Scaling Ratios — Thermally Isolated\n'
             'Cliff at 28→29q; direct-index near-2× at all steps (no cliff)')
ax.legend(loc='upper left', framealpha=0.9, fontsize=9.5)
ax.set_ylim(0, 5.5)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out = OUT_DIR / "qft_isolated_fig3_step_ratios.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")


# ── Fig 4: Cross-circuit comparison QFT vs GHZ ───────────────────────────────
fig, ax = plt.subplots(figsize=(8, 5))

bk_labels = ['C\nJAX CPU', 'F\nMLX GPU tensor', 'J\nMLX GPU direct-index', 'K\nMLX CPU direct-index']
x      = np.arange(4)
width  = 0.35

qft_cliffs = [
    next(r['mean'] for r in DATA['C'] if r['qubits'] == 29) /
    next(r['mean'] for r in DATA['C'] if r['qubits'] == 28),
    next(r['mean'] for r in DATA['F'] if r['qubits'] == 29) /
    next(r['mean'] for r in DATA['F'] if r['qubits'] == 28),
    next(r['mean'] for r in DATA['J'] if r['qubits'] == 29) /
    next(r['mean'] for r in DATA['J'] if r['qubits'] == 28),
    next(r['mean'] for r in DATA['K'] if r['qubits'] == 29) /
    next(r['mean'] for r in DATA['K'] if r['qubits'] == 28),
]
ghz_cliffs = [GHZ_CLIFF['C'], GHZ_CLIFF['F'], GHZ_CLIFF['J'], GHZ_CLIFF['K']]

bars_qft = ax.bar(x - width/2, qft_cliffs, width, label='QFT  O(n²) gates  [N=3 cross-val]',
                  color='#42A5F5', alpha=0.85, edgecolor='white')
bars_ghz = ax.bar(x + width/2, ghz_cliffs, width, label='GHZ  O(n) gates   [N=5 primary]',
                  color='#EF5350', alpha=0.85, edgecolor='white')

for bar, val in zip(bars_qft, qft_cliffs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{val:.2f}×', ha='center', va='bottom', fontsize=9, color='#1565C0', fontweight='bold')
for bar, val in zip(bars_ghz, ghz_cliffs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.04,
            f'{val:.2f}×', ha='center', va='bottom', fontsize=9, color='#B71C1C', fontweight='bold')

ax.axhline(y=2.0, color='gray', lw=1.2, ls='--', alpha=0.6, label='2× (ideal DRAM doubling)')
ax.axhline(y=4.0, color='red',  lw=1.0, ls=':',  alpha=0.35)

# Bracket direct-index backends
ax.annotate('', xy=(2.6, 0.3), xytext=(3.6, 0.3),
            arrowprops=dict(arrowstyle='<->', color='#4CAF50', lw=1.5))
ax.text(3.1, 0.5, 'direct-index: ~2.1×\ncircuit-independent', ha='center',
        fontsize=8.5, color='#2E7D32',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#E8F5E9', alpha=0.9))

ax.set_xticks(x)
ax.set_xticklabels(bk_labels, fontsize=10)
ax.set_ylabel('28→29q scaling ratio  (cliff magnitude)')
ax.set_title('DRAM Cliff: QFT vs GHZ — Circuit-Independent Memory Wall\n'
             'direct-index ~2.1× in both circuits; tensordot 3.8–4.5× regardless of gate count')
ax.legend(framealpha=0.9, loc='upper right')
ax.set_ylim(0, 5.5)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out = OUT_DIR / "qft_isolated_fig4_circuit_independence.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")

print(f"\nAll figures saved to: {OUT_DIR}")
