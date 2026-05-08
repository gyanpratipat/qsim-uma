"""
plot_qft_wall_time.py

QFT wall-time cliff chart — same style as GHZ combined chart.
Backends: C (JAX), F, J, K (MLX) — thermally isolated, N=3, 27-30q.

Data sources:
  C  — qft_C_jax.log           (N=3, first clean run)
  F  — cliff_isolated_qft_20260428_165251.csv
  J  — cliff_isolated_qft_20260428_150951.csv
  K  — cliff_isolated_qft_20260428_165251.csv
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PUB_DIR  = Path(__file__).parent.parent.parent.parent / "Publication" / "figures"
PUB_DIR.mkdir(parents=True, exist_ok=True)

# ── Data from thermally isolated logs ────────────────────────────────────────
# C: qft_C_jax.log
# F, K: qft_FK_mlx_gpu_tensor_cpu_direct_index.log (clean run; first F run had CoV>5% at 29q)
# J: qft_J_mlx_gpu_direct_index.log
DATA = {
    'C': [
        {'qubits': 27, 'mean':   49.513, 'std':  1.0996},
        {'qubits': 28, 'mean':  108.644, 'std':  0.4094},
        {'qubits': 29, 'mean':  470.793, 'std':  0.7462},
        {'qubits': 30, 'mean': 1075.165, 'std': 11.2376},
    ],
    'F': [
        {'qubits': 27, 'mean':   26.567, 'std':  0.0154},
        {'qubits': 28, 'mean':   56.459, 'std':  0.0379},
        {'qubits': 29, 'mean':  216.588, 'std':  5.9560},
        {'qubits': 30, 'mean':  466.703, 'std':  9.9215},
    ],
    'J': [
        {'qubits': 27, 'mean':   20.063, 'std':  0.2383},
        {'qubits': 28, 'mean':   42.887, 'std':  0.2795},
        {'qubits': 29, 'mean':   91.115, 'std':  0.0474},
        {'qubits': 30, 'mean':  198.578, 'std':  0.6798},
    ],
    'K': [
        {'qubits': 27, 'mean':  117.136, 'std':  0.2801},
        {'qubits': 28, 'mean':  248.952, 'std':  0.0945},
        {'qubits': 29, 'mean':  528.553, 'std':  0.1171},
        {'qubits': 30, 'mean': 1147.969, 'std':  1.2928},
    ],
}

QUBITS = [27, 28, 29, 30]

# ── Style — identical to GHZ combined chart ───────────────────────────────────
RED  = '#EF5350'
BLUE = '#42A5F5'
GRAY = '#424242'

STYLE = {
    'C': dict(color=GRAY, marker='*', ls='-',  lw=2.0, ms=12, label='C: JAX CPU tensordot',    zorder=7),
    'F': dict(color=RED,  marker='s', ls='-',  lw=2.0, ms=7,  label='F: MLX GPU tensordot',    zorder=5),
    'J': dict(color=BLUE, marker='o', ls='-',  lw=2.2, ms=7,  label='J: MLX GPU direct-index', zorder=6),
    'K': dict(color=BLUE, marker='o', ls='--', lw=1.8, ms=7,  label='K: MLX CPU direct-index', zorder=3),
}

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 9.5,
    'figure.dpi':     150,
})

fig, ax = plt.subplots(figsize=(9, 6))

for bk in ['K', 'F', 'J', 'C']:
    rows = DATA[bk]
    q = [r['qubits'] for r in rows]
    m = [r['mean']   for r in rows]
    e = [r['std']    for r in rows]
    st = STYLE[bk]
    ax.errorbar(q, m, yerr=e, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=st['ms'], capsize=4, capthick=1.2,
                label=st['label'], zorder=st['zorder'])

# Cliff shading
ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5, zorder=1)
ax.text(28.53, 0.88, 'DRAM cliff\n28→29q', fontsize=8, color='red',
        alpha=0.7, va='bottom', transform=ax.get_xaxis_transform())

ax.set_yscale('log')
ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('QFT Circuit — Wall-clock Time, Thermally Isolated (N=3)\n'
             'Solid = GPU  |  Dashed = CPU  |  Color = algorithm family')

# Legend — same layout as GHZ chart
family_patches = [
    mpatches.Patch(color=RED,  label='Tensordot  (F)'),
    mpatches.Patch(color=BLUE, label='Direct-index  (J, K)'),
    mpatches.Patch(color=GRAY, label='JAX tensordot  (C ★)'),
]
line_legend = ax.legend(loc='lower right', framealpha=0.92, ncol=1, fontsize=9)
ax.add_artist(line_legend)
ax.legend(handles=family_patches, loc='upper left', framealpha=0.92,
          fontsize=9, title='Algorithm family', title_fontsize=9)

ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.7, 30.3)

fig.tight_layout()
out = PUB_DIR / "fig2_qft_wall_time.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
