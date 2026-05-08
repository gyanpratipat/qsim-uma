"""
plot_ghz_speedup.py

GHZ GPU speedup vs STREAM-predicted 1.85× for all three algorithm families.
Data: Exp 5 (cliff_isolated_ghz_20260506_131012.csv), thermally isolated, N=5.

Pairs: (G/F) tensordot, (I/H) flat-index, (K/J) direct-index
STREAM prediction: CPU_BW/GPU_BW = 119.9/221.9 = 0.54 → GPU 1/0.54 = 1.85× faster
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

PUB_DIR = Path(__file__).parent.parent / "figures"
PUB_DIR.mkdir(parents=True, exist_ok=True)

QUBITS   = [27, 28, 29, 30]
Q_LABELS = ['27q', '28q', '29q', '30q']

# Exp 5 means (from cliff_isolated_ghz_20260506_131012.log)
MEANS = {
    'F': [1.8237, 3.7241, 11.7503, 25.2471],   # MLX GPU tensordot
    'G': [5.6374, 15.1475, 40.3363, 97.6236],   # MLX CPU tensordot
    'H': [2.0145, 4.1461, 16.7105, 35.7902],    # MLX GPU flat-index
    'I': [7.1267, 24.5551, 59.0943, 134.9402],  # MLX CPU flat-index
    'J': [1.2120, 2.5083, 5.2395, 10.9616],     # MLX GPU direct-index
    'K': [12.2163, 25.3829, 52.6155, 108.4415], # MLX CPU direct-index
}

RED   = '#EF5350'
GREEN = '#66BB6A'
BLUE  = '#42A5F5'
STREAM_PRED = 1.85

pairs = [
    ('G', 'F', 'Tensordot (G÷F)',    RED),
    ('I', 'H', 'Flat-index (I÷H)',   GREEN),
    ('K', 'J', 'Direct-index (K÷J)', BLUE),
]

x      = np.arange(len(QUBITS))
width  = 0.22
offsets = np.array([-1, 0, 1]) * width

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi':     150,
})

fig, ax = plt.subplots(figsize=(9, 5.5))

for idx, (cpu_bk, gpu_bk, lbl, col) in enumerate(pairs):
    speedups = [MEANS[cpu_bk][i] / MEANS[gpu_bk][i] for i in range(4)]
    bars = ax.bar(x + offsets[idx], speedups, width,
                  color=col, label=lbl, alpha=0.85,
                  edgecolor='white', linewidth=0.5)
    for bar, val in zip(bars, speedups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.12,
                f'{val:.2f}×', ha='center', va='bottom',
                fontsize=8.5, color=col, fontweight='bold')

ax.axhline(y=STREAM_PRED, color='black', lw=1.5, ls='--', alpha=0.7)
ax.text(3.55, STREAM_PRED + 0.12, f'{STREAM_PRED}×\nSTREAM', fontsize=8.5,
        color='black', alpha=0.7, ha='left', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(Q_LABELS, fontsize=11)
ax.set_ylabel('GPU Speedup  (CPU time ÷ GPU time)')
ax.set_title('GHZ Circuit — GPU Speedup vs STREAM Prediction (N=5, Thermally Isolated)\n'
             'Direct-index ~10× throughout; tensordot & flat-index exceed prediction pre-cliff')
ax.legend(loc='upper right', framealpha=0.92, fontsize=10, ncol=4)
ax.set_ylim(0, 13.5)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out = PUB_DIR / "fig3_ghz_speedup.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
