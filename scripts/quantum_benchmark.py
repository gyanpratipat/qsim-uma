"""
Quantum Simulation Memory Wall — Multi-Trial Benchmark Suite
Author: Anon Capybara
Purpose: Paper-ready statistical benchmarking of quantum circuit simulation
         approaches from a Computer Architecture and Memory Systems perspective

Algorithms tested:
  A. Brute Force NumPy   — O(4^n) memory, full Kronecker expansion
  B. pykronecker         — O(2^n × k) memory, lazy Kronecker
  C. JAX tensordot       — O(2^n) memory, XLA-compiled direct ops
  D. direct-index reconstruction — O(2^n) memory, pure index manipulation
  E. External SSD        — storage-centric, state vector on NVMe

Statistical methodology:
  - 1 warm-up run per (algorithm, qubit_count) — discarded
  - N_TRIALS = 7 timed runs
  - Metrics: mean, std, min, max, median wall-clock time
  - Memory: peak allocation via tracemalloc (Python heap)
  - Separate JIT compilation time from steady-state time

Memory hierarchy context (M4 Pro):
  L1 cache:  192 KB  → fits ~12k complex64 elements → useful to ~13q
  L2 cache:  16 MB   → fits ~1M complex64 elements  → useful to ~19q
  L3 cache:  24 MB   → fits ~1.5M complex64 elements→ useful to ~20q
  DRAM:      48 GB   → fits ~3B complex64 elements  → useful to ~30q
  NVMe:      ~2 TB   → theoretically to ~37q+
"""

import os
os.environ['JAX_PLATFORMS'] = 'cpu'   # must be set before JAX is imported anywhere

import numpy as np
import tracemalloc
import time
import json
import sys
import statistics
import functools
import multiprocessing as mp
from pathlib import Path

# ── Config ──────────────────────────────────────────────────
N_TRIALS    = 7          # timed runs per data point
TIMEOUT_S   = 180        # skip if single run exceeds this

# Guard against re-execution in spawn worker subprocesses
if mp.current_process().name == 'MainProcess':
    _EXP_TS  = time.strftime("%Y%m%d_%H%M%S")
    RESULTS_DIR = Path(__file__).parent.parent / "results" / f"exp_{_EXP_TS}"
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for _i in range(1, N_TRIALS + 1):
        (RESULTS_DIR / f"run_{_i}").mkdir(exist_ok=True)
    (RESULTS_DIR / "inference").mkdir(exist_ok=True)
else:
    RESULTS_DIR = None

# Qubit ranges per algorithm
QUBITS_BRUTE   = list(range(3, 20))    # brute force hits wall early
QUBITS_OPT     = list(range(3, 31))    # optimised approaches
QUBITS_SSD     = [30, 31, 32, 33]      # storage-centric only
QUBITS_MLX     = list(range(3, 33))    # MLX: tensor reaches 31q, flat hits int32 at 32q

# Memory hierarchy breakpoints for M4 Pro (state vector in complex64, 8 bytes)
# q  →  2^q * 8 bytes
CACHE_BREAKPOINTS = {
    'L1 (192KB)': 14,   # 2^14 * 8 = 131KB — fits in L1
    'L2 (16MB)':  20,   # 2^20 * 8 = 8MB  — fits in L2
    'L3 (24MB)':  21,   # 2^21 * 8 = 16MB — fits in L3
    'DRAM':       29,   # 2^29 * 8 = 4GB  — DRAM territory
}

# ── Utility ─────────────────────────────────────────────────
def state_vector_gb(n_qubits):
    """Theoretical state vector size in GB (complex64 = 8 bytes)"""
    return (2**n_qubits * 8) / (1024**3)

def time_trial(fn, *args, timeout=TIMEOUT_S):
    """
    Run fn(*args) once and return (elapsed_s, peak_mem_gb).
    Returns (None, None) on OOM or timeout.
    """
    tracemalloc.start()
    t0 = time.perf_counter()
    try:
        fn(*args)
        elapsed = time.perf_counter() - t0
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if elapsed > timeout:
            return None, None
        return elapsed, peak / (1024**3)
    except (MemoryError, Exception) as e:
        tracemalloc.stop()
        print(f"    ✗ {type(e).__name__}: {e}")
        return None, None

def _isolated_worker(fn, args):
    """Worker executed in a fresh subprocess — returns (elapsed_s, mem_gb).
    Must be a module-level function so the spawn context can pickle it.

    prepare() is called before the timer so that MLX import + Metal driver
    initialisation are excluded from the measured wall-clock time.
    mem_gb_theoretical() is used for MLX backends because tracemalloc only
    captures Python heap; MLX uses its own device allocator invisble to it.
    """
    import tracemalloc, time
    # Warm up backend (import, driver init) BEFORE starting the clock.
    if hasattr(fn, 'prepare'):
        fn.prepare()
    tracemalloc.start()
    t0 = time.perf_counter()
    fn(*args)
    elapsed = time.perf_counter() - t0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    # For MLX backends tracemalloc returns ~0; use theoretical state-vector size.
    if hasattr(fn, 'mem_gb_theoretical'):
        mem_gb = fn.mem_gb_theoretical(args[0])   # args[0] is always n (qubits)
    else:
        mem_gb = peak / (1024**3)
    return elapsed, mem_gb


def time_trial_isolated(fn, *args, timeout=TIMEOUT_S):
    """
    Run fn(*args) in a one-shot subprocess so fatal C++ crashes (e.g. MLX
    Metal int32 overflow at 31q) don't kill the main process.
    Spawn context re-imports the module cleanly; fn must be picklable
    (use module-level _MLXGhz class, not closures).
    Returns (elapsed_s, peak_mem_gb) or (None, None) on crash/timeout.
    """
    ctx = mp.get_context('spawn')
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        try:
            r = pool.apply_async(_isolated_worker, (fn, args))
            elapsed, mem = r.get(timeout=timeout + 15)
            if elapsed > timeout:
                pool.terminate()
                return None, None
            return elapsed, mem
        except mp.TimeoutError:
            pool.terminate()
            print(f"    ✗ Subprocess timeout (>{timeout}s)")
            return None, None
        except Exception as e:
            print(f"    ✗ Subprocess crashed: {type(e).__name__}: {e}")
            return None, None


def run_trials(label, fn, qubit_list, args_fn, isolate=False):
    """
    For each qubit count:
      1 warm-up → N_TRIALS timed runs → statistics

    isolate=True: each trial runs in a fresh subprocess via time_trial_isolated,
    so fatal C++ exceptions (MLX Metal OOM/int32 overflow) don't kill main.

    Returns:
      inference_results  — list of aggregated stat dicts (mean/std/median/…)
      per_run_results    — list of N_TRIALS lists, each containing raw
                           {algorithm, qubits, trial, time, mem_gb} dicts
                           for writing to run_1/ … run_N/ folders
    """
    _trial = time_trial_isolated if isolate else time_trial

    inference_results = []
    per_run_results   = [[] for _ in range(N_TRIALS)]

    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")

    for n in qubit_list:
        args  = args_fn(n)
        sv_gb = state_vector_gb(n)
        print(f"\n  {n:2d} qubits  [state vector: {sv_gb:.4f} GB]")

        # Warm-up (discard — handles JIT compilation and cache priming)
        print(f"    warm-up...", end='', flush=True)
        t, m = _trial(fn, *args)
        if t is None:
            print(f" failed/timeout — skipping remaining qubits")
            break
        print(f" {t:.3f}s (discarded)")

        # Timed trials
        times, mems = [], []
        failed = False
        for trial in range(N_TRIALS):
            t, m = _trial(fn, *args)
            if t is None:
                failed = True
                break
            times.append(t)
            mems.append(m)
            print(f"    trial {trial+1}/{N_TRIALS}: {t:.4f}s  {m:.6f} GB")
            per_run_results[trial].append({
                'algorithm':      label,
                'qubits':         n,
                'state_vector_gb': sv_gb,
                'trial':          trial + 1,
                'time':           t,
                'mem_gb':         m,
            })

        if failed or not times:
            print(f"    → Trial failed — stopping this algorithm at {n}q")
            break

        result = {
            'algorithm':      label,
            'qubits':         n,
            'state_vector_gb': sv_gb,
            'time_mean':      statistics.mean(times),
            'time_std':       statistics.stdev(times) if len(times) > 1 else 0,
            'time_min':       min(times),
            'time_max':       max(times),
            'time_median':    statistics.median(times),
            'mem_mean_gb':    statistics.mean(mems),
            'mem_max_gb':     max(mems),
            'n_trials':       len(times),
        }
        inference_results.append(result)
        print(f"    → mean={result['time_mean']:.4f}s  "
              f"std={result['time_std']:.4f}s  "
              f"mem_peak={result['mem_max_gb']:.4f} GB")

    return inference_results, per_run_results


# ── Algorithm A: Brute Force NumPy ──────────────────────────
def brute_force_ghz(n):
    """
    Naive full Kronecker expansion.
    Memory: O(4^n) — builds 2^n × 2^n gate matrix.
    This is what author's original 2022 code did.
    """
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    
    h = np.array([[1,1],[1,-1]], dtype=np.complex64) / np.sqrt(2)
    I = np.eye(2, dtype=np.complex64)
    cnot_2q = np.array([[1,0,0,0],[0,1,0,0],
                         [0,0,0,1],[0,0,1,0]], dtype=np.complex64)
    
    def apply_single(state, gate, qubit, n_q):
        left  = np.eye(2**qubit, dtype=np.complex64)
        right = np.eye(2**(n_q - qubit - 1), dtype=np.complex64)
        G = np.kron(np.kron(left, gate), right)   # ← O(4^n) allocation
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            return G @ state

    # H on qubit 0
    state = apply_single(state, h, 0, n)
    # CNOT cascade
    for i in range(n - 1):
        left  = np.eye(2**i, dtype=np.complex64)
        right = np.eye(2**(n - i - 2), dtype=np.complex64)
        G = np.kron(np.kron(left, cnot_2q), right)
        with np.errstate(divide='ignore', over='ignore', invalid='ignore'):
            state = G @ state
    return state


# ── Algorithm D: direct-index Reconstruction ─────────────────────────
def direct_index_h_gate(state, target, n_q, idx=None):
    """
    direct-index-style Hadamard: direct index manipulation, no Kronecker.
    Selects exactly the 2^(n-1) pairs affected by the gate.
    Memory: O(2^n) — only the state vector, no temporaries.

    idx: precomputed arange(2^n) — pass from direct_index_ghz to avoid
    reallocating 2^n × 8 bytes on every gate call (27 allocs at 30q = 216 GB).
    """
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2))
    if idx is None:
        idx = np.arange(len(state), dtype=np.int64)

    # Little-endian: qubit i = bit position i (matches Qiskit)
    mask0 = ((idx >> target) & 1) == 0
    mask1 = ~mask0

    a = state[mask0]
    b = state[mask1]
    state[mask0] = inv_sqrt2 * (a + b)
    state[mask1] = inv_sqrt2 * (a - b)
    return state

def direct_index_cnot_gate(state, ctrl, tgt, n_q, idx=None):
    """
    direct-index-style CNOT: direct index manipulation.
    Only touches 2^(n-2) element pairs where ctrl=1.
    """
    if idx is None:
        idx = np.arange(len(state), dtype=np.int64)

    ctrl_bit = ctrl   # little-endian: qubit i = bit position i
    tgt_bit  = tgt

    ctrl_mask = ((idx >> ctrl_bit) & 1) == 1
    tgt0_mask = ((idx >> tgt_bit)  & 1) == 0
    swap_mask = ctrl_mask & tgt0_mask

    swap_idx = idx[swap_mask]
    partner  = swap_idx ^ (1 << tgt_bit)

    tmp = state[swap_idx].copy()
    state[swap_idx] = state[partner]
    state[partner]  = tmp
    return state

def direct_index_ghz(n):
    """Full GHZ circuit using direct-index-style direct index manipulation."""
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    idx = np.arange(2**n, dtype=np.int64)   # precomputed once; reused across all gates

    state = direct_index_h_gate(state, 0, n, idx)
    for i in range(n - 1):
        state = direct_index_cnot_gate(state, i, i+1, n, idx)
    return state


# ── Algorithm B: pykronecker ─────────────────────────────────
def pykronecker_ghz(n):
    """
    Lazy Kronecker via pykronecker library.
    Never materialises the full 2^n × 2^n matrix.
    Memory: O(2^n) state vector + small overhead per gate.
    This is what author's V3 used before JAX.
    """
    try:
        from pykronecker import KroneckerProduct as kp
    except ImportError:
        raise ImportError("pykronecker not installed: pip install pykronecker")
    
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    
    h = np.array([[1,1],[1,-1]], dtype=np.complex64) / np.sqrt(2)
    I = np.eye(2, dtype=np.complex64)
    cnot_2q = np.array([[1,0,0,0],[0,1,0,0],
                         [0,0,0,1],[0,0,1,0]], dtype=np.complex64)
    
    def apply_lazy(state, gate, qubit, n_q):
        n_gate_q = int(np.log2(gate.shape[0]))
        ops = [I]*qubit + [gate] + [I]*(n_q - qubit - n_gate_q)
        G = kp(ops)
        return np.array(G @ state)
    
    state = apply_lazy(state, h, 0, n)
    for i in range(n - 1):
        state = apply_lazy(state, cnot_2q, i, n)
    return state


# ── Algorithm C: JAX tensordot ───────────────────────────────
def jax_ghz_factory():
    """
    JAX-compiled GHZ using tensordot (XLA backend).
    JIT-compiled on first call — warm-up handles this.
    Memory: O(2^n) — XLA eliminates intermediate allocations.
    Note: tracemalloc does NOT capture JAX device memory.
    Actual memory = same O(2^n) as direct-index — just not visible to Python heap.
    """
    try:
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_platform_name', 'cpu')
    except ImportError:
        raise ImportError("jax not installed: pip install jax")
    
    h_mat  = jnp.array([[1,1],[1,-1]], dtype=jnp.complex64) / jnp.sqrt(2)
    cnot_m = jnp.array([[1,0,0,0],[0,1,0,0],
                          [0,0,0,1],[0,0,1,0]], dtype=jnp.complex64)
    
    # qubit and n_q are static: Python list ops ([2]*n_q, axes[:qubit])
    # cannot be traced by JAX — mark them static so XLA recompiles per value.
    @functools.partial(jax.jit, static_argnums=(2, 3))
    def apply_gate_jax(state, gate, qubit, n_q):
        n_gate = int(np.log2(gate.shape[0]))
        state_t = state.reshape([2]*n_q)
        result = jnp.tensordot(gate.reshape([2]*(n_gate*2)),
                                state_t,
                                axes=([n_gate + k for k in range(n_gate)],
                                      [qubit + k for k in range(n_gate)]))
        axes = list(range(n_gate, n_q))
        axes = axes[:qubit] + list(range(n_gate)) + axes[qubit:]
        result = jnp.transpose(result, axes)
        return result.reshape(2**n_q)
    
    def ghz(n):
        state = jnp.zeros(2**n, dtype=jnp.complex64)
        state = state.at[0].set(1.0)
        state = apply_gate_jax(state, h_mat, 0, n)
        for i in range(n-1):
            state = apply_gate_jax(state, cnot_m, i, n)
        return np.array(state)  # back to numpy
    
    return ghz


# ── Algorithm F/G/H/I: MLX GPU/CPU, Tensor/Flat ─────────────

class _MLXGhz:
    """
    Picklable MLX GHZ runner used by time_trial_isolated().
    Closures from mlx_ghz_factory() can't cross process boundaries;
    module-level class instances can.
    MLX is imported lazily inside __call__ so the main process never
    touches the Metal driver — only the subprocess does.
    """
    def __init__(self, device, representation):
        self.device = device                  # 'gpu' or 'cpu' — picklable string
        self.representation = representation  # 'tensor' or 'flat'

    def prepare(self):
        """Pre-import MLX and init the device before the timed section starts.
        Called by _isolated_worker so that import + driver-init overhead is
        excluded from the measured wall-clock time.
        mx.eval forces the Metal command queue to actually initialise, not
        just the Python binding — without it, first-kernel overhead leaks in.
        """
        import mlx.core as mx
        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)
        _ = mx.zeros(1, dtype=mx.complex64)
        mx.eval(_)

    def mem_gb_theoretical(self, n):
        """tracemalloc is blind to MLX device memory (uses its own allocator).
        Return the theoretical state-vector size so the benchmark reports a
        meaningful number instead of ~0 GB.
        """
        return (2**n * 8) / (1024**3)   # complex64: 8 bytes per element

    def __call__(self, n):
        import mlx.core as mx
        import numpy as _np

        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)

        _h_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
        _cnot_np = _np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_np.complex64)
        H    = mx.array(_h_np)
        CNOT = mx.array(_cnot_np)

        if self.representation == 'tensor':
            state = mx.zeros([2]*n, dtype=mx.complex64)
            state[tuple([0]*n)] = 1.0

            def apply_gate(state_t, gate, qi):
                nq = len(state_t.shape)
                if gate.shape[0] == 2:
                    psi = mx.tensordot(gate, state_t, axes=[[1], [qi]])
                    order = list(range(1, qi+1)) + [0] + list(range(qi+1, nq))
                    return mx.transpose(psi, order)
                else:
                    g = gate.reshape(2, 2, 2, 2)
                    psi = mx.tensordot(g, state_t, axes=[[2,3], [qi, qi+1]])
                    order = list(range(2, qi+2)) + [0, 1] + list(range(qi+2, nq))
                    return mx.transpose(psi, order)

            state = apply_gate(state, H, 0)
            for i in range(n - 1):
                state = apply_gate(state, CNOT, i)
        else:
            state = mx.zeros(2**n, dtype=mx.complex64)
            state[0] = 1.0

            def apply_gate_flat(state_f, gate, qi, nq):
                state_t = state_f.reshape([2]*nq)
                if gate.shape[0] == 2:
                    psi = mx.tensordot(gate, state_t, axes=[[1], [qi]])
                    order = list(range(1, qi+1)) + [0] + list(range(qi+1, nq))
                    psi = mx.transpose(psi, order)
                else:
                    g = gate.reshape(2, 2, 2, 2)
                    psi = mx.tensordot(g, state_t, axes=[[2,3], [qi, qi+1]])
                    order = list(range(2, qi+2)) + [0, 1] + list(range(qi+2, nq))
                    psi = mx.transpose(psi, order)
                return psi.reshape([2**nq])

            state = apply_gate_flat(state, H, 0, n)
            for i in range(n - 1):
                state = apply_gate_flat(state, CNOT, i, n)

        mx.eval(state)
        return _np.array(state.flatten()[:8])


def mlx_ghz_factory(device='gpu', representation='tensor'):
    """
    MLX GHZ via tensordot — gate logic ported directly from bench_gpu.py.

    device:         'gpu' (Metal) or 'cpu'
    representation: 'tensor' — state kept as [2]*n throughout, bypassing
                               MLX's int32 element-count ceiling (→ 31q max)
                    'flat'   — state kept as 2**n; reshape inside apply_gate
                               overflows int32 at 32q (same as bench_gpu.py)

    mx.eval() forces the lazy graph to complete before timing ends.
    tracemalloc captures ~0 for MLX device memory (same caveat as JAX).
    """
    try:
        import mlx.core as mx
    except ImportError:
        raise ImportError("mlx not installed: pip install mlx")

    dev = mx.gpu if device == 'gpu' else mx.cpu

    _h_np   = np.array([[1,1],[1,-1]], dtype=np.complex64) / np.sqrt(2)
    _cnot_np = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex64)

    if representation == 'tensor':
        def ghz(n):
            mx.set_default_device(dev)
            state = mx.zeros([2]*n, dtype=mx.complex64)
            state[tuple([0]*n)] = 1.0
            H    = mx.array(_h_np)
            CNOT = mx.array(_cnot_np)

            def apply_gate(state_t, gate, qubit_index):
                nq = len(state_t.shape)
                if gate.shape[0] == 2:
                    psi = mx.tensordot(gate, state_t, axes=[[1], [qubit_index]])
                    order = list(range(1, qubit_index+1)) + [0] + list(range(qubit_index+1, nq))
                    return mx.transpose(psi, order)
                else:
                    g = gate.reshape(2, 2, 2, 2)
                    psi = mx.tensordot(g, state_t, axes=[[2,3], [qubit_index, qubit_index+1]])
                    order = (list(range(2, qubit_index+2)) + [0, 1] +
                             list(range(qubit_index+2, nq)))
                    return mx.transpose(psi, order)

            state = apply_gate(state, H, 0)
            for i in range(n - 1):
                state = apply_gate(state, CNOT, i)
            mx.eval(state)
            return np.array(state.flatten()[:8])
    else:
        def ghz(n):
            mx.set_default_device(dev)
            state = mx.zeros(2**n, dtype=mx.complex64)
            state[0] = 1.0
            H    = mx.array(_h_np)
            CNOT = mx.array(_cnot_np)

            def apply_gate_flat(state_f, gate, qubit_index, nq):
                state_t = state_f.reshape([2]*nq)
                if gate.shape[0] == 2:
                    psi = mx.tensordot(gate, state_t, axes=[[1], [qubit_index]])
                    order = list(range(1, qubit_index+1)) + [0] + list(range(qubit_index+1, nq))
                    psi = mx.transpose(psi, order)
                else:
                    g = gate.reshape(2, 2, 2, 2)
                    psi = mx.tensordot(g, state_t, axes=[[2,3], [qubit_index, qubit_index+1]])
                    order = (list(range(2, qubit_index+2)) + [0, 1] +
                             list(range(qubit_index+2, nq)))
                    psi = mx.transpose(psi, order)
                return psi.reshape([2**nq])

            state = apply_gate_flat(state, H, 0, n)
            for i in range(n - 1):
                state = apply_gate_flat(state, CNOT, i, n)
            mx.eval(state)
            return np.array(state.flatten()[:8])

    return ghz


# ── Algorithm J/K: MLX direct-index — direct index manipulation ──────
class _MLXDirectIndexGhz:
    """
    Picklable MLX direct-index GHZ runner — direct boolean index manipulation,
    no tensordot, no reshape. Implements the QARN paper Section II algorithm
    ported to MLX arrays. Little-endian convention: qubit i = bit position i.

    Compared to _MLXGhz (tensordot): same bandwidth demand, different
    compute pattern. The gap between them quantifies algorithmic cost of
    tensordot vs direct scatter/gather, independent of hardware.

    idx precomputed once per circuit — avoids ~n × 2^n byte re-allocs.
    mx.where creates a new array each gate call (MLX is lazy/functional)
    so both mask=0 and mask=1 paths are expressed in one pass cleanly.
    """
    def __init__(self, device):
        self.device = device   # 'gpu' or 'cpu'

    def prepare(self):
        """Pre-import MLX and warm up device before timed section."""
        import mlx.core as mx
        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)
        _ = mx.zeros(1, dtype=mx.complex64)
        mx.eval(_)

    def mem_gb_theoretical(self, n):
        """tracemalloc is blind to MLX device memory; return theoretical size."""
        return (2**n * 8) / (1024**3)

    def __call__(self, n):
        import mlx.core as mx
        import numpy as _np

        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)

        # List syntax avoids MLX scalar-int → int32 cast that overflows at 2^31.
        # int32 idx is fine up to 30q (max index 2^30-1 fits); 31q+ will ERR at arange.
        # Per-gate mx.eval keeps graph small: lazy 29-step graph is ~2x slower at 30q.
        state = mx.zeros([2**n], dtype=mx.complex64)
        state[0] = 1.0
        idx = mx.arange(2**n, dtype=mx.int32)   # precomputed once
        mx.eval(state, idx)

        # H gate on qubit 0 (little-endian: qubit 0 = bit 0)
        inv_sqrt2 = float(_np.float32(1.0 / _np.sqrt(2)))
        partner = idx ^ 1
        is_zero = (idx & 1) == 0
        pv = state[partner]
        state = mx.where(is_zero,
                         inv_sqrt2 * (state + pv),
                         inv_sqrt2 * (pv - state))
        mx.eval(state)

        # CNOT cascade: ctrl=i, tgt=i+1
        for i in range(n - 1):
            partner = idx ^ (1 << (i + 1))
            state   = mx.where(((idx >> i) & 1) == 1, state[partner], state)
            mx.eval(state)

        return _np.array(state[:8])


# ── Algorithm E: External SSD ────────────────────────────────
def ssd_ghz_factory(ssd_path="/tmp/qbench_ssd"):
    """
    Storage-centric simulation: state vector stored on NVMe.
    Uses memory-mapped files — OS manages paging between SSD and RAM.
    Theoretical ceiling: limited by SSD capacity, not RAM.
    Memory: O(chunk_size) RAM + O(2^n) on disk.
    
    This implements the 'storage-centric' approach mentioned in
    QARN's future work section.
    """
    os.makedirs(ssd_path, exist_ok=True)
    
    def ghz(n):
        sv_bytes = 2**n * 8  # complex64
        sv_path  = os.path.join(ssd_path, f"state_{n}q.bin")
        
        # Create memory-mapped state vector on disk
        sv = np.memmap(sv_path, dtype=np.complex64, mode='w+', shape=(2**n,))
        sv[0] = 1.0
        
        h_mat   = np.array([[1,1],[1,-1]], dtype=np.complex64) / np.sqrt(2)
        inv_sq2 = np.float32(1.0 / np.sqrt(2))
        
        # direct-index-style gate on memory-mapped array
        # Key insight: memmap reads/writes go through OS page cache
        # → bandwidth limited by NVMe, not DRAM
        
        idx = np.arange(2**n, dtype=np.int64)
        
        # H on qubit 0 — little-endian: qubit 0 = bit 0 (matches direct-index convention)
        bit_pos = 0
        mask0 = ((idx >> bit_pos) & 1) == 0
        mask1 = ~mask0
        a = np.array(sv[mask0])  # explicit copy from disk to RAM chunk
        b = np.array(sv[mask1])
        sv[mask0] = inv_sq2 * (a + b)
        sv[mask1] = inv_sq2 * (a - b)
        sv.flush()

        # CNOT cascade — little-endian: qubit i = bit i
        for ctrl in range(n-1):
            tgt      = ctrl + 1
            ctrl_bit = ctrl
            tgt_bit  = tgt
            ctrl_mask = ((idx >> ctrl_bit) & 1) == 1
            tgt0_mask = ((idx >> tgt_bit)  & 1) == 0
            swap_mask = ctrl_mask & tgt0_mask
            swap_idx  = idx[swap_mask]
            partner   = swap_idx ^ (1 << tgt_bit)
            
            tmp = np.array(sv[swap_idx])
            sv[swap_idx] = sv[partner]
            sv[partner]  = tmp
            sv.flush()
        
        result = np.array(sv[:8])  # read first 8 to verify
        del sv
        os.remove(sv_path)  # clean up
        return result
    
    return ghz


# ── MLX int32 overflow test ───────────────────────────────────
def test_mlx_int32_fix():
    """
    Test if passing a list bypasses MLX's int32 array size limit.
    MLX uses int32 internally for array indexing.
    2^31 = 2,147,483,648 elements overflows int32.
    
    Potential fix: use Python list syntax or dtype override.
    """
    try:
        import mlx.core as mx
        print("\n── MLX int32 overflow test ──────────────────")
        
        # Test 1: Standard call (known to overflow at 31q)
        try:
            x = mx.zeros(2**30)  # 30q — should work (1GB)
            mx.eval(x)
            print(f"  mx.zeros(2**30): ✓ shape={x.shape}")
        except Exception as e:
            print(f"  mx.zeros(2**30): ✗ {e}")
        
        # Test 2: List syntax — does it bypass int32?
        try:
            x = mx.zeros([2**30])
            mx.eval(x)
            print(f"  mx.zeros([2**30]): ✓ shape={x.shape}")
        except Exception as e:
            print(f"  mx.zeros([2**30]): ✗ {e}")
        
        # Test 3: Tuple syntax
        try:
            x = mx.zeros((2**30,))
            mx.eval(x)
            print(f"  mx.zeros((2**30,)): ✓ shape={x.shape}")
        except Exception as e:
            print(f"  mx.zeros((2**30,)): ✗ {e}")
        
        # Test 4: 31q with list
        try:
            print("  Testing 31q (2^31 = 2147483648 elements)...")
            x = mx.zeros([2**31])
            mx.eval(x)
            print(f"  mx.zeros([2**31]): ✓ shape={x.shape} — int32 fix works!")
        except Exception as e:
            print(f"  mx.zeros([2**31]): ✗ {type(e).__name__}: {e}")
            print("  → int32 is a library-level limit, not a Python syntax issue")
        
        # Test 5: Chunked approach as workaround
        print("\n  Chunked workaround for 31q+:")
        print("  → Split state vector into 2 chunks of 2^30")
        print("  → Process each chunk separately")
        print("  → Only valid for separable gate operations")
        print("  → For entangling gates (CNOT), chunks must communicate")
        print("  → This is essentially distributed simulation")
        
    except ImportError:
        print("  MLX not available in this environment")


# ── Memory hierarchy analysis ─────────────────────────────────
def print_memory_hierarchy():
    """
    Print the memory hierarchy context for M4 Pro.
    This frames every benchmark in Comp Arch terms.
    """
    print("\n" + "="*60)
    print("  MEMORY HIERARCHY CONTEXT — Apple M4 Pro")
    print("="*60)
    print(f"  {'Level':<12} {'Size':<12} {'BW (GB/s)':<12} {'Max Qubits':<12} {'State Vector'}")
    print(f"  {'-'*60}")
    
    hierarchy = [
        ("Registers", "~1 KB",    "N/A",    3,  "trivial"),
        ("L1 Cache",  "192 KB",   "~3000",  14, "131 KB"),
        ("L2 Cache",  "16 MB",    "~2000",  20, "8 MB"),
        ("L3 Cache",  "24 MB",    "~1500",  21, "16 MB"),
        ("DRAM",      "48 GB",    "106",    30, "8 GB"),
        ("NVMe SSD",  "~2 TB",    "~3.5",   37, "1 TB"),
    ]
    
    for level, size, bw, max_q, sv in hierarchy:
        marker = " ← DRAM cliff" if level == "DRAM" else ""
        print(f"  {level:<12} {size:<12} {bw:<12} {max_q:<12} {sv}{marker}")
    
    print("\n  Key insight: Each tier transition changes the dominant bottleneck")
    print("  ≤14q: L1 cache-resident → compute-bound (very fast)")
    print("  15-20q: L2/L3 territory → cache bandwidth bound")
    print("  21-28q: DRAM territory → memory bandwidth bound")
    print("  29-30q: DRAM capacity edge → page fault / swap cliff")
    print("  31q+:  NVMe territory → storage bandwidth bound")


# ── Arithmetic intensity analysis ─────────────────────────────
def print_arithmetic_intensity():
    """
    Roofline model analysis for each gate type and algorithm.
    Arithmetic intensity = FLOP / byte transferred
    """
    print("\n" + "="*60)
    print("  ARITHMETIC INTENSITY ANALYSIS (Roofline Model)")
    print("="*60)
    print("  M4 Pro peaks: ~3.6 TFLOP/s, ~106 GB/s (JAX CPU)")
    print("  Ridge point: 3600/106 ≈ 34 FLOP/byte")
    print("  → Operations below 34 FLOP/byte are MEMORY-BOUND")
    print()
    print(f"  {'Algorithm':<25} {'FLOPs/elem':<12} {'Bytes/elem':<12} {'AI (F/B)':<10} {'Bound'}")
    print(f"  {'-'*65}")
    
    ops = [
        # name, flops_per_element_pair, bytes_per_element_pair
        ("Brute Force (kron+matmul)", 8,    32,  "Memory"),
        ("  — H gate component",     8,    32,  "Memory"),
        ("pykronecker (lazy)",        8,    32,  "Memory"),
        ("direct-index (index manip.)",       6,    32,  "Memory"),
        ("  — H gate",               6,    32,  "Memory"),
        ("  — CNOT gate",            0,    32,  "Memory (swap only)"),
        ("  — X gate",               0,    32,  "Memory (swap only)"),
    ]
    
    for name, flops, bytes_, bound in ops:
        ai = flops / bytes_ if bytes_ > 0 else float('inf')
        print(f"  {name:<25} {flops:<12} {bytes_:<12} {ai:<10.3f} {bound}")
    
    print()
    print("  All quantum gate operations have AI << 34 FLOP/byte")
    print("  → Quantum simulation is ALWAYS memory-bound on M4 Pro")
    print("  → Faster hardware helps only if it has more bandwidth")


# ── C++ projection ────────────────────────────────────────────
def print_cpp_projection():
    """
    Project C++ performance from Python measurements.
    This contextualises why QARN moved to Cython/CUDA.
    """
    print("\n" + "="*60)
    print("  C++ PERFORMANCE PROJECTION")
    print("="*60)
    print()
    print("  Current Python (NumPy/JAX) overhead sources:")
    print("  1. Python interpreter: ~100ns per function call")
    print("  2. NumPy array bounds checking: ~10-20% overhead")
    print("  3. GIL: prevents true multi-threading")
    print("  4. Object allocation: Python ints/floats are heap objects")
    print()
    print("  Expected C++ speedups (same algorithm, same memory pattern):")
    print(f"  {'Regime':<25} {'Python':<15} {'C++ (projected)':<20} {'Speedup'}")
    print(f"  {'-'*65}")
    print(f"  {'≤20q (cache-resident)':<25} {'baseline':<15} {'3-5× faster':<20} {'compute-bound'}")
    print(f"  {'21-28q (DRAM-bound)':<25} {'baseline':<15} {'1.2-1.5× faster':<20} {'memory-bound'}")
    print(f"  {'29-30q (bandwidth cliff)':<25} {'baseline':<15} {'1.1-1.2× faster':<20} {'IO-bound'}")
    print()
    print("  Key insight: In the bandwidth-bound regime (21q+),")
    print("  C++ provides minimal benefit because the bottleneck")
    print("  is memory bandwidth, not instruction execution speed.")
    print()
    print("  This is why QARN's Cython layer + CUDA matters:")
    print("  → Cython eliminates Python overhead in cache-resident ops")
    print("  → CUDA provides 10× bandwidth (not 10× compute)")
    print("  → The memory wall just moves from 30q to 35q")


# ── Main runner ───────────────────────────────────────────────
def main():
    print("\n" + "="*60)
    print("  QUANTUM SIMULATION MEMORY WALL BENCHMARK")
    print("  Multi-Trial Statistical Study")
    print(f"  N_TRIALS = {N_TRIALS} per data point")
    print("="*60)
    
    # Print theoretical analysis first
    print_memory_hierarchy()
    print_arithmetic_intensity()
    print_cpp_projection()
    
    # Test MLX int32 fix
    test_mlx_int32_fix()
    
    all_inference = []
    all_per_run   = [[] for _ in range(N_TRIALS)]

    def _collect(inf, prun):
        all_inference.extend(inf)
        for i, run_data in enumerate(prun):
            all_per_run[i].extend(run_data)
        # Incremental checkpoint — survives a crash mid-run
        ckpt = RESULTS_DIR / "inference" / "checkpoint.json"
        with open(ckpt, 'w') as f:
            json.dump(all_inference, f, indent=2)

    # ── A: Brute Force ──
    print("\n[A] Brute Force NumPy — O(4^n) memory")
    _collect(*run_trials("A_BruteForce_NumPy", brute_force_ghz, QUBITS_BRUTE, lambda n: (n,)))

    # ── D: direct-index Reconstruction ──
    print("\n[D] direct-index Reconstruction — O(2^n) direct index manipulation")
    _collect(*run_trials("D_direct-index_Reconstruction", direct_index_ghz, QUBITS_OPT, lambda n: (n,)))

    # ── B: pykronecker ──
    print("\n[B] pykronecker — O(2^n) lazy Kronecker")
    try:
        _collect(*run_trials("B_pykronecker", pykronecker_ghz, QUBITS_OPT, lambda n: (n,)))
    except ImportError as e:
        print(f"  Skipping pykronecker: {e}")

    # ── C: JAX ──
    print("\n[C] JAX tensordot — O(2^n) XLA-compiled")
    try:
        jax_ghz = jax_ghz_factory()
        _collect(*run_trials("C_JAX_tensordot", jax_ghz, QUBITS_OPT, lambda n: (n,)))
    except ImportError as e:
        print(f"  Skipping JAX: {e}")

    # ── F: MLX GPU tensor ──
    # _MLXGhz is picklable (module-level class) — required for isolate=True.
    # isolate=True spawns a fresh subprocess per trial so the fatal C++ Metal
    # OOM crash at 31q+ does not kill the main process.
    print("\n[F] MLX GPU (Metal, tensor [2]*n) — bypasses int32 ceiling")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("F_MLX_GPU_Tensor", _MLXGhz('gpu', 'tensor'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX GPU tensor: {e}")

    # ── G: MLX CPU tensor ──
    print("\n[G] MLX CPU (tensor [2]*n)")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("G_MLX_CPU_Tensor", _MLXGhz('cpu', 'tensor'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX CPU tensor: {e}")

    # ── H: MLX GPU flat ──
    print("\n[H] MLX GPU flat (2^n) — int32 ceiling hits at 32q")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("H_MLX_GPU_Flat", _MLXGhz('gpu', 'flat'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX GPU flat: {e}")

    # ── I: MLX CPU flat ──
    print("\n[I] MLX CPU flat (2^n) — int32 ceiling hits at 32q")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("I_MLX_CPU_Flat", _MLXGhz('cpu', 'flat'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX CPU flat: {e}")

    # ── J: MLX GPU direct-index (direct index manipulation) ──
    # Key comparison: tensordot (F) vs direct-index (J) on same GPU hardware.
    # Gap quantifies algorithmic cost of tensordot vs direct scatter/gather,
    # bandwidth-normalised — directly comparable to QARN L40 result.
    print("\n[J] MLX GPU direct-index — direct boolean index manipulation, no tensordot")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("J_MLX_GPU_direct-index", _MLXDirectIndexGhz('gpu'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX GPU direct-index: {e}")

    # ── K: MLX CPU direct-index (direct index manipulation) ──
    print("\n[K] MLX CPU direct-index — direct boolean index manipulation, no tensordot")
    try:
        import mlx.core as _mlx_check
        _collect(*run_trials("K_MLX_CPU_direct-index", _MLXDirectIndexGhz('cpu'),
                             QUBITS_MLX, lambda n: (n,), isolate=True))
    except ImportError as e:
        print(f"  Skipping MLX CPU direct-index: {e}")

    # ── E: External SSD ──
    print("\n[E] External SSD — storage-centric simulation")
    _collect(*run_trials("E_External_SSD", ssd_ghz_factory(),
                         [25, 27, 28, 29, 30], lambda n: (n,)))

    # ── Save per-run raw timings ──
    for i, run_data in enumerate(all_per_run):
        run_file = RESULTS_DIR / f"run_{i+1}" / "results.json"
        with open(run_file, 'w') as f:
            json.dump(run_data, f, indent=2)

    # ── Save aggregated inference results ──
    inf_file = RESULTS_DIR / "inference" / "results.json"
    with open(inf_file, 'w') as f:
        json.dump(all_inference, f, indent=2)

    print(f"\n\nExperiment saved to {RESULTS_DIR}/")
    print(f"  ├─ run_1/ … run_{N_TRIALS}/   (raw per-trial timings)")
    print(f"  └─ inference/                  (aggregated mean/std/median)")

    # ── Summary table ──
    print("\n" + "="*60)
    print("  SUMMARY TABLE")
    print("="*60)
    print(f"\n  {'Algorithm':<28} {'Qubits':<8} {'Mean(s)':<10} {'Std':<8} {'Mem(GB)':<10}")
    print(f"  {'-'*65}")
    for r in all_inference:
        print(f"  {r['algorithm']:<28} {r['qubits']:<8} "
              f"{r['time_mean']:<10.4f} {r['time_std']:<8.4f} "
              f"{r['mem_max_gb']:<10.6f}")

    return all_inference


if __name__ == "__main__":
    mp.freeze_support()   # required for spawn context on packaged macOS apps
    results = main()
