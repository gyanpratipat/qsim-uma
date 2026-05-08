"""
QFT (Quantum Fourier Transform) benchmark — 10 backends (A–K).
Mirrors the GHZ benchmark in quantum_benchmark.py for the QFT circuit.

QFT circuit (n qubits):
  for j in 0..n-1: H(j), CP(2pi/2^(k-j+1), j, k) for k in j+1..n-1
  bit-reversal: SWAP(i, n-1-i) for i in 0..n//2-1
  Total gates: n H + n(n-1)/2 CP + n//2 SWAP  ≈ n^2/2  (O(n^2))

Backends:
  A  NumPy tensordot    plain numpy, no JIT/GPU (brute-force baseline)
  B  pykronecker        lazy Kronecker for H; numpy tensordot for non-adj CP/SWAP
  C  JAX CPU            XLA-compiled tensordot
  D  NumPy direct-index         in-place bitmask ops (QARN algorithm), no framework
  E  SSD memmap direct-index    state vector on NVMe via np.memmap
  F  MLX GPU tensor     tensordot on [2]*n tensor, Metal GPU
  G  MLX CPU tensor     tensordot on [2]*n tensor, Apple AMX
  H  MLX GPU flat       reshape to [2]*n per gate, Metal GPU
  I  MLX CPU flat       reshape to [2]*n per gate, Apple AMX
  J  MLX GPU direct-index       bitmask ops on flat array, Metal GPU
  K  MLX CPU direct-index       bitmask ops on flat array, Apple AMX

Paper questions:
  Q1: Does the 29q DRAM cliff appear in QFT as in GHZ? (GHZ: direct-index 2.09x, JAX 4.30x)
  Q2: GPU/CPU speedup for QFT — same 1.64x as GHZ or different?
  Q3: MLX GPU direct-index 30q time — how does O(n^2) gate count affect vs QARN GHZ baseline?
"""

import numpy as np
import math
import time
import sys
import os
import datetime

os.environ['JAX_PLATFORMS'] = 'cpu'

import jax.numpy as jnp
import mlx.core as mx


# =========================================================================
# Logging
# =========================================================================
class Tee:
    def __init__(self, filename):
        self.file = open(filename, 'w', buffering=1)
        self.stdout = sys.stdout
        sys.stdout = self

    def write(self, data):
        self.stdout.write(data)
        self.file.write(data)

    def flush(self):
        self.stdout.flush()
        self.file.flush()

    def close(self):
        sys.stdout = self.stdout
        self.file.close()


timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
tee = Tee(f'qft_bench_{timestamp}.log')


# =========================================================================
# Gate matrices
# =========================================================================
H_NP    = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
SWAP_NP = np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64)

def cp_np(theta):
    return np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                     [0,0,0,np.exp(1j * np.float64(theta))]], dtype=np.complex64)


# =========================================================================
# Shared helpers — NumPy (used by A and B)
# =========================================================================
def _apply_1q_np(s, gate, q):
    psi = np.tensordot(gate, s, axes=[[1], [q]])
    return np.moveaxis(psi, 0, q)

def _apply_2q_np(s, gate4, q1, q2):
    g = gate4.reshape(2, 2, 2, 2)
    psi = np.tensordot(g, s, axes=[[2, 3], [q1, q2]])
    return np.moveaxis(psi, [0, 1], [q1, q2])


# =========================================================================
# Shared helpers — JAX (C)
# =========================================================================
def _apply_1q_jax(s, gate, q):
    psi = jnp.tensordot(jnp.array(gate), s, axes=[[1], [q]])
    return jnp.moveaxis(psi, 0, q)

def _apply_2q_jax(s, gate4, q1, q2):
    g = jnp.array(gate4).reshape(2, 2, 2, 2)
    psi = jnp.tensordot(g, s, axes=[[2, 3], [q1, q2]])
    return jnp.moveaxis(psi, [0, 1], [q1, q2])


# =========================================================================
# Shared helpers — MLX tensor (F/G/H/I)
# =========================================================================
def _apply_1q_mlx(s, gate, q):
    n = len(s.shape)
    psi = mx.tensordot(gate, s, axes=[[1], [q]])
    axes_order = list(range(1, q+1)) + [0] + list(range(q+1, n))
    return mx.transpose(psi, axes_order)

def _apply_2q_mlx(s, gate4, q1, q2):
    """Non-adjacent 2-qubit gate on tensor state.
    axes_order derived from: after tensordot(g, s, axes=[[2,3],[q1,q2]]),
    psi axes are [out_q1, out_q2, remaining qubits in original order].
    """
    n = len(s.shape)
    g = gate4.reshape(2, 2, 2, 2)
    psi = mx.tensordot(g, s, axes=[[2, 3], [q1, q2]])
    axes_order = (list(range(2, q1+2)) + [0] +
                  list(range(q1+2, q2+1)) + [1] +
                  list(range(q2+1, n)))
    return mx.transpose(psi, axes_order)


# =========================================================================
# Shared helpers — NumPy direct-index (D/E)
# =========================================================================
def _direct_index_h_np(state, target, idx):
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2))
    mask0 = ((idx >> target) & 1) == 0
    mask1 = ~mask0
    a = state[mask0].copy()
    b = state[mask1].copy()
    state[mask0] = inv_sqrt2 * (a + b)
    state[mask1] = inv_sqrt2 * (a - b)
    return state

def _direct_index_cp_np(state, q1, q2, theta, idx):
    mask = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 1)
    state[mask] *= np.complex64(np.exp(1j * np.float64(theta)))
    return state

def _direct_index_swap_np(state, q1, q2, idx):
    # Process only pairs where q1=1, q2=0 (each pair touched once)
    swap_mask = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 0)
    swap_idx = idx[swap_mask]
    partner  = swap_idx ^ (1 << q1) ^ (1 << q2)
    tmp = state[swap_idx].copy()
    state[swap_idx] = state[partner]
    state[partner]  = tmp
    return state


# =========================================================================
# Shared helpers — MLX direct-index (J/K)
# =========================================================================
def _direct_index_h_mlx(s, target, idx):
    inv_sqrt2 = float(np.float32(1.0 / np.sqrt(2)))
    partner   = idx ^ (1 << target)
    is_zero   = ((idx >> target) & 1) == 0
    pv        = s[partner]
    return mx.where(is_zero, inv_sqrt2*(s + pv), inv_sqrt2*(pv - s))

def _direct_index_cp_mlx(s, q1, q2, theta, idx):
    mask  = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 1)
    phase = complex(np.exp(1j * np.float64(theta)))
    return mx.where(mask, s * phase, s)

def _direct_index_swap_mlx(s, q1, q2, idx):
    differs = ((idx >> q1) & 1) != ((idx >> q2) & 1)
    partner = idx ^ (1 << q1) ^ (1 << q2)
    return mx.where(differs, s[partner], s)


# =========================================================================
# A — NumPy brute force  (full Kronecker expansion per gate, O(4^n) memory)
#     Identical in approach to GHZ backend A in quantum_benchmark.py.
#     Non-adjacent 2-qubit gates (CP, bit-reversal SWAP) are decomposed via
#     the SWAP trick: bubble qubit q2 left to q1+1, apply gate, bubble back.
#     Each SWAP and gate application builds a fresh 2^n × 2^n matrix.
#     Expected OOM at ~15q, same as GHZ backend A.
# =========================================================================
def qft_brute_force(n):
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0

    def apply_1q(s, gate, q):
        left  = np.eye(2**q,       dtype=np.complex64)
        right = np.eye(2**(n-q-1), dtype=np.complex64)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return np.kron(np.kron(left, gate), right) @ s

    def apply_2q_adj(s, gate4, q):   # gate on adjacent qubits (q, q+1)
        left  = np.eye(2**q,       dtype=np.complex64)
        right = np.eye(2**(n-q-2), dtype=np.complex64)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return np.kron(np.kron(left, gate4.reshape(4,4)), right) @ s

    def apply_2q(s, gate4, q1, q2):  # q1 < q2, possibly non-adjacent
        # Bubble qubit q2 left to position q1+1 via adjacent SWAPs
        for p in range(q2-1, q1, -1):
            s = apply_2q_adj(s, SWAP_NP, p)
        s = apply_2q_adj(s, gate4, q1)
        # Reverse: restore qubit back to original position q2
        for p in range(q1+1, q2):
            s = apply_2q_adj(s, SWAP_NP, p)
        return s

    for j in range(n):
        state = apply_1q(state, H_NP, j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            state = apply_2q(state, cp_np(theta), j, k)

    for i in range(n // 2):
        state = apply_2q(state, SWAP_NP, i, n-1-i)

    return state


# =========================================================================
# B — pykronecker
#   pykronecker handles 1-qubit (H) and adjacent 2-qubit gates via lazy kron.
#   QFT CP gates are non-adjacent → fall back to numpy tensordot for those.
# =========================================================================
def qft_pykronecker(n):
    try:
        from pykronecker import KroneckerProduct as kp
    except ImportError:
        raise ImportError("pykronecker not installed: pip install pykronecker")

    I = np.eye(2, dtype=np.complex64)
    s = np.zeros(2**n, dtype=np.complex64)
    s[0] = 1.0

    for j in range(n):
        # H via lazy Kronecker
        ops = [I]*j + [H_NP] + [I]*(n-j-1)
        s = np.array(kp(ops) @ s)
        # CP gates: non-adjacent → numpy tensordot on tensor view
        st = s.reshape([2]*n)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            st = _apply_2q_np(st, cp_np(theta), j, k)
        s = st.flatten()

    # SWAP via numpy tensordot
    st = s.reshape([2]*n)
    for i in range(n // 2):
        st = _apply_2q_np(st, SWAP_NP, i, n-1-i)
    return st.flatten()


# =========================================================================
# C — JAX CPU  (XLA-compiled tensordot)
# =========================================================================
def qft_jax(n):
    s = jnp.zeros([2]*n, dtype=jnp.complex64)
    s = s.at[tuple([0]*n)].set(1.0)
    H    = jnp.array(H_NP)
    SWAP = jnp.array(SWAP_NP)
    for j in range(n):
        s = _apply_1q_jax(s, H, j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = _apply_2q_jax(s, cp_np(theta), j, k)
    for i in range(n // 2):
        s = _apply_2q_jax(s, SWAP, i, n-1-i)
    s.block_until_ready()
    return s


# =========================================================================
# D — NumPy direct-index  (QARN algorithm, in-place bitmask, no framework)
# =========================================================================
def qft_numpy_direct_index(n):
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    idx = np.arange(2**n, dtype=np.int64)
    for j in range(n):
        state = _direct_index_h_np(state, j, idx)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            state = _direct_index_cp_np(state, j, k, theta, idx)
    for i in range(n // 2):
        state = _direct_index_swap_np(state, i, n-1-i, idx)
    return state


# =========================================================================
# E — SSD memmap direct-index  (state vector on NVMe via np.memmap)
# =========================================================================
_SSD_PATH = "/tmp/qbench_qft_ssd"

def qft_ssd(n):
    os.makedirs(_SSD_PATH, exist_ok=True)
    sv_path = os.path.join(_SSD_PATH, f"state_{n}q.bin")
    sv  = np.memmap(sv_path, dtype=np.complex64, mode='w+', shape=(2**n,))
    sv[0] = 1.0
    idx = np.arange(2**n, dtype=np.int64)

    inv_sqrt2 = np.float32(1.0 / np.sqrt(2))
    for j in range(n):
        # H gate
        mask0 = ((idx >> j) & 1) == 0
        mask1 = ~mask0
        a = np.array(sv[mask0]); b = np.array(sv[mask1])
        sv[mask0] = inv_sqrt2 * (a + b)
        sv[mask1] = inv_sqrt2 * (a - b)
        sv.flush()
        # CP gates
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            phase = np.complex64(np.exp(1j * np.float64(theta)))
            mask = (((idx >> j) & 1) == 1) & (((idx >> k) & 1) == 1)
            sv[mask] = np.array(sv[mask]) * phase
            sv.flush()

    # SWAP gates
    for i in range(n // 2):
        q1, q2 = i, n-1-i
        swap_mask = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 0)
        swap_idx  = idx[swap_mask]
        partner   = swap_idx ^ (1 << q1) ^ (1 << q2)
        tmp = np.array(sv[swap_idx])
        sv[swap_idx] = sv[partner]
        sv[partner]  = tmp
        sv.flush()

    result = np.array(sv[:8])
    del sv
    try:
        os.remove(sv_path)
    except OSError:
        pass
    return result


# =========================================================================
# F/G — MLX GPU/CPU tensor  (tensordot, [2]*n state, row-wise eval)
# =========================================================================
def qft_mlx(n, device):
    mx.set_default_device(device)
    s    = mx.zeros([2]*n, dtype=mx.complex64)
    s[tuple([0]*n)] = 1.0
    H    = mx.array(H_NP)
    SWAP = mx.array(SWAP_NP)
    for j in range(n):
        s = _apply_1q_mlx(s, H, j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = _apply_2q_mlx(s, mx.array(cp_np(theta)), j, k)
        mx.eval(s)   # flush lazy graph after each qubit's H+CP row
    for i in range(n // 2):
        s = _apply_2q_mlx(s, SWAP, i, n-1-i)
    mx.eval(s)
    return s


# =========================================================================
# H/I — MLX GPU/CPU flat  (reshape to [2]*n per gate, state as 1D)
# =========================================================================
def _apply_1q_mlx_flat(s, gate, q, n):
    st = s.reshape([2]*n)
    st = _apply_1q_mlx(st, gate, q)
    return st.reshape([2**n])

def _apply_2q_mlx_flat(s, gate4, q1, q2, n):
    st = s.reshape([2]*n)
    st = _apply_2q_mlx(st, gate4, q1, q2)
    return st.reshape([2**n])

def qft_mlx_flat(n, device):
    mx.set_default_device(device)
    s    = mx.zeros(2**n, dtype=mx.complex64)
    s[0] = 1.0
    H    = mx.array(H_NP)
    SWAP = mx.array(SWAP_NP)
    for j in range(n):
        s = _apply_1q_mlx_flat(s, H, j, n)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = _apply_2q_mlx_flat(s, mx.array(cp_np(theta)), j, k, n)
        mx.eval(s)
    for i in range(n // 2):
        s = _apply_2q_mlx_flat(s, SWAP, i, n-1-i, n)
    mx.eval(s)
    return s


# =========================================================================
# J/K — MLX GPU/CPU direct-index  (flat array, bitmask, per-gate eval)
# =========================================================================
def qft_mlx_direct_index(n, device):
    mx.set_default_device(device)
    s   = mx.zeros([2**n], dtype=mx.complex64)
    s[0] = 1.0
    idx = mx.arange(2**n, dtype=mx.int32)   # int32 safe to 30q; 31q overflows
    mx.eval(s, idx)
    for j in range(n):
        s = _direct_index_h_mlx(s, j, idx)
        mx.eval(s)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = _direct_index_cp_mlx(s, j, k, theta, idx)
            mx.eval(s)
    for i in range(n // 2):
        s = _direct_index_swap_mlx(s, i, n-1-i, idx)
        mx.eval(s)
    return s


# =========================================================================
# Qiskit verification
# =========================================================================
def verify_qft(n):
    """Cross-check all key backends against Qiskit Aer at small n."""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
    except ImportError as e:
        print(f"    SKIP — qiskit not available: {e}")
        return True

    qc = QuantumCircuit(n)
    for j in range(n):
        qc.h(j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            qc.cp(theta, j, k)
    for i in range(n // 2):
        qc.swap(i, n-1-i)
    qc.save_statevector()
    ref = np.abs(np.array(
        AerSimulator(method='statevector').run(qc).result().get_statevector()
    ))

    checks = [
        ('A brute',    np.array(qft_brute_force(n)).flatten()),
        ('C JAX',      np.array(qft_jax(n)).flatten()),
        ('D direct-index-np',  np.array(qft_numpy_direct_index(n)).flatten()),
        ('F MLX-GPU',  np.array(qft_mlx(n, mx.gpu)).flatten()),
        ('J direct-index-GPU', np.array(qft_mlx_direct_index(n, mx.gpu)).flatten()),
    ]
    all_ok = True
    for label, sv in checks:
        ok = np.allclose(np.abs(sv), ref, atol=1e-5)
        print(f"    {label:12s} {n}q: {'PASS' if ok else 'FAIL'}")
        if not ok:
            all_ok = False
            print(f"      expected |amp|≈{1/math.sqrt(2**n):.5f}, "
                  f"got max={np.abs(sv).max():.5f}")
    return all_ok


# =========================================================================
# Effective bandwidth
# =========================================================================
def effective_bw(n_qubits, n_gates, passes_per_gate, elapsed_s):
    """
    Compute effective memory bandwidth utilization.

    passes_per_gate:
      direct-index:      2 (read pair + write pair, full state each gate)
      tensordot: 3 (read + contract + transpose)
      CNOT direct-index: 1 (quarter state only)
    """
    bytes_total = (2**n_qubits * 8) * n_gates * passes_per_gate
    return bytes_total / elapsed_s / 1e9  # GB/s


def effective_bw_qft(n_qubits, elapsed_s):
    """
    QFT-specific effective bandwidth using per-gate-type data movement.
    Models minimum necessary bytes: direct-index-style access patterns.

    H:    full state, 2 passes  (read + write, 2^n elements)
    CP:   quarter state, 2 passes  (only |11> elements: 2^(n-2))
    SWAP: quarter state, 4 passes  (read×2 + write×2 for each swapped pair)
    """
    n_h    = n_qubits
    n_cp   = n_qubits * (n_qubits - 1) // 2
    n_swap = n_qubits // 2

    bytes_h    = n_h    * 2**n_qubits     * 8 * 2
    bytes_cp   = n_cp   * 2**(n_qubits-2) * 8 * 2
    bytes_swap = n_swap * 2**(n_qubits-2) * 8 * 4

    total_bytes = bytes_h + bytes_cp + bytes_swap
    return total_bytes / elapsed_s / 1e9


# =========================================================================
# Bandwidth probe
# =========================================================================
def measure_bw(name, fn_array, fn_eval, size_mb=512):
    n = int(size_mb * 1e6 / 4)
    a = fn_array(n)
    fn_eval(a)
    t0 = time.perf_counter()
    for _ in range(5):
        b = a * 2.0
        fn_eval(b)
    t1 = time.perf_counter()
    bw = (5 * 2 * size_mb) / ((t1 - t0) * 1e3)
    print(f"  {name}: {bw:.1f} GB/s")


# =========================================================================
# Benchmark runner
# =========================================================================
def bench(name, fn, warmup_n=4, max_q=30, timeout=600):
    print(f"\n=== {name} ===")
    print(f"{'qubits':>6} {'time(s)':>10} {'n_gates':>8} {'state_GB':>10} {'eff_GB/s':>10}  status")
    try:
        fn(warmup_n); fn(warmup_n + 1)
    except Exception as e:
        print(f"  warmup failed: {e} — skipping backend")
        return
    for n in range(3, max_q + 1):
        n_gates  = n + n*(n-1)//2 + n//2
        state_gb = (2**n * 8) / 1e9
        try:
            t0      = time.perf_counter()
            _       = fn(n)
            elapsed = time.perf_counter() - t0
            eff_bw  = effective_bw_qft(n, elapsed)
            print(f"{n:>6} {elapsed:>10.4f} {n_gates:>8} {state_gb:>10.4f} {eff_bw:>10.2f}  ok")
            sys.stdout.flush()
            if elapsed > timeout:
                print(f"  >> timeout at {n}q ({elapsed:.1f}s > {timeout}s)")
                break
        except Exception as e:
            print(f"{n:>6} {'---':>10} {n_gates:>8} {state_gb:>10.4f} {'---':>10}  ERR: {e}")
            sys.stdout.flush()
            break


# =========================================================================
# Main
# =========================================================================
print("=" * 64)
print(f"  QFT Benchmark (10 backends)  —  {datetime.datetime.now()}")
print("=" * 64)
print("Circuit:  H + CP + SWAP  (O(n^2) gates)\n")

print("--- Qiskit verification (3q, 4q) ---")
all_ok = True
for nv in (3, 4):
    ok = verify_qft(nv)
    all_ok = all_ok and ok

if not all_ok:
    print("\nVERIFICATION FAILED — aborting.")
    tee.close()
    sys.exit(1)
print("  All verifications PASSED\n")

# Warm up large-array kernels on all three backends before probing bandwidth
print("--- Pre-probe warmup (20q QFT on each backend) ---")
print("  JAX CPU ...", end=" ", flush=True); qft_jax(20); print("done")
print("  MLX GPU ...", end=" ", flush=True); qft_mlx(20, mx.gpu); print("done")
print("  MLX CPU ...", end=" ", flush=True); qft_mlx(20, mx.cpu); print("done")
print()

print("--- Bandwidth probe (post-warmup) ---")
measure_bw("JAX CPU ", lambda n: jnp.ones(n, dtype=jnp.float32),
           lambda x: x.block_until_ready())
mx.set_default_device(mx.gpu)
measure_bw("MLX GPU ", lambda n: mx.ones(n, dtype=mx.float32),
           lambda x: mx.eval(x))
mx.set_default_device(mx.cpu)
measure_bw("MLX CPU ", lambda n: mx.ones(n, dtype=mx.float32),
           lambda x: mx.eval(x))
print()

bench("A  NumPy brute force (Kronecker)",
      qft_brute_force,
      warmup_n=4, max_q=20, timeout=600)

bench("B  pykronecker",
      qft_pykronecker,
      warmup_n=4, max_q=30, timeout=600)

bench("C  JAX CPU (XLA tensordot)",
      qft_jax,
      warmup_n=4, max_q=30, timeout=2000)

bench("D  NumPy direct-index (bitmask)",
      qft_numpy_direct_index,
      warmup_n=4, max_q=30, timeout=600)

bench("E  SSD memmap direct-index",
      qft_ssd,
      warmup_n=4, max_q=30, timeout=600)

bench("F  MLX GPU tensor (Metal)",
      lambda n: qft_mlx(n, mx.gpu),
      warmup_n=4, max_q=30, timeout=600)

bench("G  MLX CPU tensor",
      lambda n: qft_mlx(n, mx.cpu),
      warmup_n=4, max_q=30, timeout=600)

bench("H  MLX GPU flat (Metal)",
      lambda n: qft_mlx_flat(n, mx.gpu),
      warmup_n=4, max_q=30, timeout=600)

bench("I  MLX CPU flat",
      lambda n: qft_mlx_flat(n, mx.cpu),
      warmup_n=4, max_q=30, timeout=600)

bench("J  MLX GPU direct-index (Metal)",
      lambda n: qft_mlx_direct_index(n, mx.gpu),
      warmup_n=4, max_q=30, timeout=600)

bench("K  MLX CPU direct-index",
      lambda n: qft_mlx_direct_index(n, mx.cpu),
      warmup_n=4, max_q=30, timeout=600)

print(f"\n{'='*64}")
print(f"  Done — {datetime.datetime.now()}")
print(f"  Log: qft_bench_{timestamp}.log")
print(f"{'='*64}")

tee.close()
