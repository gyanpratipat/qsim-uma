"""
plot_ghz_combined.py

Combined 7-backend GHZ wall-time cliff chart.
C from Exp 2 (cliff_isolated_ghz_20260428_094411.csv)
F,G,H,I,J,K from Exp 5 (cliff_isolated_ghz_20260506_131012.csv)
Both thermally isolated, N=5.
"""

import csv
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent.parent / "plots"
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXP2_CSV = Path(__file__).parent.parent.parent / "2_ghz_isolated" / "data" / "cliff_isolated_ghz_20260428_094411.csv"
EXP5_CSV = Path(__file__).parent.parent / "data" / "cliff_isolated_ghz_20260506_131012.csv"

def load_backend(path, backend_key):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            if row['backend'] == backend_key:
                rows.append({'qubits': int(row['qubits']),
                             'mean':   float(row['mean_s']),
                             'std':    float(row['std_s'])})
    return sorted(rows, key=lambda r: r['qubits'])

DATA = {
    'C': load_backend(EXP2_CSV, 'C'),
    'F': load_backend(EXP5_CSV, 'F'),
    'G': load_backend(EXP5_CSV, 'G'),
    'H': load_backend(EXP5_CSV, 'H'),
    'I': load_backend(EXP5_CSV, 'I'),
    'J': load_backend(EXP5_CSV, 'J'),
    'K': load_backend(EXP5_CSV, 'K'),
}

QUBITS = [27, 28, 29, 30]

# 3 primary colors — one per algorithm family; GPU=solid, CPU=dashed
# C is the only JAX backend — gray, star marker, labelled explicitly
RED   = '#EF5350'   # tensordot family (F, G)
GREEN = '#66BB6A'   # flat-index family (H, I)
BLUE  = '#42A5F5'   # direct-index family (J, K)
GRAY  = '#424242'   # C — JAX (standalone)

STYLE = {
    'C': dict(color=GRAY,  marker='*', ls='-',  lw=2.0, ms=12, label='C: JAX CPU tensordot', zorder=7),
    'F': dict(color=RED,   marker='s', ls='-',  lw=2.0, ms=7,  label='F: MLX GPU tensordot',        zorder=5),
    'G': dict(color=RED,   marker='s', ls='--', lw=1.8, ms=7,  label='G: MLX CPU tensordot',        zorder=4),
    'H': dict(color=GREEN, marker='^', ls='-',  lw=2.0, ms=7,  label='H: MLX GPU flat-index',       zorder=5),
    'I': dict(color=GREEN, marker='^', ls='--', lw=1.8, ms=7,  label='I: MLX CPU flat-index',       zorder=4),
    'J': dict(color=BLUE,  marker='o', ls='-',  lw=2.2, ms=7,  label='J: MLX GPU direct-index',     zorder=6),
    'K': dict(color=BLUE,  marker='o', ls='--', lw=1.8, ms=7,  label='K: MLX CPU direct-index',     zorder=3),
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

for bk in ['K', 'G', 'I', 'F', 'H', 'J', 'C']:   # draw C and J/H on top
    rows = DATA[bk]
    q = [r['qubits'] for r in rows]
    m = [r['mean']   for r in rows]
    e = [r['std']    for r in rows]
    st = STYLE[bk]
    ax.errorbar(q, m, yerr=e, color=st['color'], marker=st['marker'],
                ls=st['ls'], lw=st['lw'], ms=st['ms'], capsize=4, capthick=1.2,
                label=st['label'], zorder=st['zorder'])


# Cliff shading and line
ax.axvspan(27.9, 29.1, alpha=0.07, color='red', zorder=0)
ax.axvline(x=28.5, color='red', lw=1.0, ls=':', alpha=0.5, zorder=1)
ax.text(28.53, 0.88, 'DRAM cliff\n28→29q', fontsize=8, color='red',
        alpha=0.7, va='bottom', transform=ax.get_xaxis_transform())

ax.set_yscale('log')
ax.set_xticks(QUBITS)
ax.set_xticklabels([f'{n}q\n({2**n*8/1e9:.2f} GB)' for n in QUBITS])
ax.set_xlabel('Qubit count  (state vector size)')
ax.set_ylabel('Wall-clock time (s)  [log scale]')
ax.set_title('GHZ Circuit — Wall-clock Time, All Backends, Thermally Isolated (N=5)\n'
             'Solid = GPU  |  Dashed = CPU  |  Color = algorithm family')


import matplotlib.patches as mpatches
family_patches = [
    mpatches.Patch(color=RED,   label='Tensordot  (F, G)'),
    mpatches.Patch(color=GREEN, label='Flat-index  (H, I)'),
    mpatches.Patch(color=BLUE,  label='Direct-index  (J, K)'),
    mpatches.Patch(color=GRAY,  label='JAX tensordot  (C ★)'),
]
# Individual backend lines — lower right
line_legend = ax.legend(loc='lower right', framealpha=0.92, ncol=1, fontsize=9)
ax.add_artist(line_legend)
# Algorithm family colors — upper left
ax.legend(handles=family_patches, loc='upper left', framealpha=0.92,
          fontsize=9, title='Algorithm family', title_fontsize=9)

ax.grid(True, which='both', alpha=0.25)
ax.set_xlim(26.7, 30.3)


fig.tight_layout()
out = OUT_DIR / "fig_ghz_all_backends.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
