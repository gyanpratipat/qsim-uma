"""
plot_qft_speedup.py

QFT GPU speedup vs STREAM-predicted 1.85× — two algorithm pairs.

Pair 1 — Tensordot (cross-framework): C (JAX CPU) ÷ F (MLX GPU tensor)
  Note: different frameworks; shows real-world JAX-CPU vs MLX-GPU advantage.

Pair 2 — Direct-index (within-framework): K (MLX CPU) ÷ J (MLX GPU)
  Same framework; isolates hardware (CPU vs GPU) from software.

Data source: experiments/4_qft_isolated/
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

PUB_DIR = Path(__file__).parent.parent / "figures"
PUB_DIR.mkdir(parents=True, exist_ok=True)

QUBITS   = [27, 28, 29, 30]
Q_LABELS = ['27q', '28q', '29q', '30q']

# From qft_C_jax.log
C_MEANS = [49.513, 108.644, 470.793, 1075.165]
# From qft_FK log
F_MEANS = [26.567,  56.459, 216.588,  466.703]
# From qft_J_mlx_gpu_direct_index.log
J_MEANS = [20.063,  42.887,  91.115,  198.578]
# From qft_FK log (K column)
K_MEANS = [117.136, 248.952, 528.553, 1147.969]

CF_SPEEDUP = [C_MEANS[i] / F_MEANS[i] for i in range(4)]   # [1.86, 1.92, 2.17, 2.30]
KJ_SPEEDUP = [K_MEANS[i] / J_MEANS[i] for i in range(4)]   # [5.84, 5.80, 5.80, 5.78]

STREAM_PRED = 1.85
RED  = '#EF5350'
BLUE = '#42A5F5'

x      = np.arange(len(QUBITS))
width  = 0.3
offsets = np.array([-0.5, 0.5]) * width

plt.rcParams.update({
    'font.family':    'sans-serif',
    'font.size':      11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'legend.fontsize': 10,
    'figure.dpi':     150,
})

fig, ax = plt.subplots(figsize=(9, 5.5))

# Tensordot cross-framework: C÷F
bars_cf = ax.bar(x + offsets[0], CF_SPEEDUP, width,
                 color=RED, alpha=0.85, edgecolor='white',
                 label='Tensordot  C÷F  (JAX CPU ÷ MLX GPU) [cross-framework]')
for bar, val in zip(bars_cf, CF_SPEEDUP):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
            f'{val:.2f}×', ha='center', va='bottom',
            fontsize=9, color=RED, fontweight='bold')

# Direct-index within-framework: K÷J
bars_kj = ax.bar(x + offsets[1], KJ_SPEEDUP, width,
                 color=BLUE, alpha=0.85, edgecolor='white',
                 label='Direct-index  K÷J  (MLX CPU ÷ MLX GPU) [same framework]')
for bar, val in zip(bars_kj, KJ_SPEEDUP):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.08,
            f'{val:.2f}×', ha='center', va='bottom',
            fontsize=9, color=BLUE, fontweight='bold')

ax.axhline(y=STREAM_PRED, color='black', lw=1.5, ls='--', alpha=0.7,
           label=f'STREAM prediction: {STREAM_PRED}×  (CPU 119.9 / GPU 221.9 GB/s)')
ax.text(3.65, STREAM_PRED + 0.08, f'{STREAM_PRED}×\nSTREAM', fontsize=8.5,
        color='black', alpha=0.7, ha='left', va='bottom')

ax.set_xticks(x)
ax.set_xticklabels(Q_LABELS, fontsize=11)
ax.set_ylabel('GPU Speedup  (CPU time ÷ GPU time)')
ax.set_title('QFT Circuit — GPU Speedup vs STREAM Prediction (N=3, Thermally Isolated)\n'
             'Direct-index ~5.8× far exceeds STREAM; tensordot ~1.9-2.3× near prediction')
ax.legend(loc='upper left', framealpha=0.92, fontsize=9.5)
ax.set_ylim(0, 8.0)
ax.grid(True, axis='y', alpha=0.25)

fig.tight_layout()
out = PUB_DIR / "fig4_qft_speedup.png"
fig.savefig(out, dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Saved: {out}")
