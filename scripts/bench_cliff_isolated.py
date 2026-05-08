"""
bench_cliff_isolated.py

Thermally isolated cliff-region benchmark (27–30q, 4 backends).
Addresses thermal artifact in quantum_benchmark.py where JAX ran after hours
of sequential A+B execution, inflating 29q time ~2.7×.

Run each circuit separately:
  caffeinate -i python3 bench_cliff_isolated.py --circuit ghz
  caffeinate -i python3 bench_cliff_isolated.py --circuit qft

Do NOT combine circuits in one invocation (thermal cross-contamination).
QFT results are labeled 'cross-validation, N=3' — do not average with GHZ N=5.
"""

import os, sys, time, math, statistics, functools, argparse, csv
import subprocess, multiprocessing as mp
import numpy as np
from pathlib import Path
from datetime import datetime

os.environ['JAX_PLATFORMS'] = 'cpu'   # must precede jax import

# ── Config ────────────────────────────────────────────────────────────────────
QUBITS       = [27, 28, 29, 30]
N_TRIALS_GHZ = 5
N_TRIALS_QFT = 3
COOL_SECS    = 90      # sleep between backends
TIMEOUT_GHZ  = 120     # s per trial (generous for 30q GHZ ≈ 40s)
TIMEOUT_QFT  = 2400    # s per trial (generous for 30q QFT ≈ 1037s JAX)

BACKENDS = {
    'C': 'JAX CPU tensordot',
    'F': 'MLX GPU tensor',
    'G': 'MLX CPU tensor',
    'H': 'MLX GPU flat-index',
    'I': 'MLX CPU flat-index',
    'J': 'MLX GPU direct-index',
    'K': 'MLX CPU direct-index',
}

# ── Module-level subprocess worker (must be picklable) ────────────────────────
def _isolated_worker(fn, args):
    """Called inside a fresh spawn subprocess. prepare() excluded from timing."""
    import time as _time
    if hasattr(fn, 'prepare'):
        fn.prepare()
    t0 = _time.perf_counter()
    fn(*args)
    elapsed = _time.perf_counter() - t0
    return elapsed


def _run_isolated(fn, n, timeout_s):
    """Run fn(n) in one-shot subprocess. Returns elapsed_s or None on failure."""
    ctx = mp.get_context('spawn')
    with ctx.Pool(1, maxtasksperchild=1) as pool:
        try:
            r = pool.apply_async(_isolated_worker, (fn, (n,)))
            return r.get(timeout=timeout_s + 30)
        except mp.TimeoutError:
            pool.terminate()
            return None
        except Exception as e:
            print(f"    [subprocess crashed: {type(e).__name__}: {e}]", flush=True)
            return None


# ── MLX GHZ subprocess runner ─────────────────────────────────────────────────
class _MLXGhzRunner:
    def __init__(self, device, variant):
        self.device  = device    # 'gpu' or 'cpu'
        self.variant = variant   # 'tensor', 'flat', or 'direct_index'

    def prepare(self):
        import mlx.core as mx
        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)
        mx.eval(mx.zeros(1, dtype=mx.complex64))

    def __call__(self, n):
        import mlx.core as mx
        import numpy as _np

        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)

        h_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
        cnot_np = _np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=_np.complex64)
        H, CNOT = mx.array(h_np), mx.array(cnot_np)

        if self.variant == 'tensor':
            s = mx.zeros([2]*n, dtype=mx.complex64)
            s[tuple([0]*n)] = 1.0

            def apply_gate(s, gate, qi):
                nq = len(s.shape)
                if gate.shape[0] == 2:
                    psi   = mx.tensordot(gate, s, axes=[[1], [qi]])
                    order = list(range(1, qi+1)) + [0] + list(range(qi+1, nq))
                else:
                    g     = gate.reshape(2, 2, 2, 2)
                    psi   = mx.tensordot(g, s, axes=[[2, 3], [qi, qi+1]])
                    order = list(range(2, qi+2)) + [0, 1] + list(range(qi+2, nq))
                return mx.transpose(psi, order)

            s = apply_gate(s, H, 0)
            for i in range(n - 1):
                s = apply_gate(s, CNOT, i)
            mx.eval(s)

        elif self.variant == 'flat':
            s = mx.zeros(2**n, dtype=mx.complex64)
            s[0] = 1.0

            def apply_gate_flat(s, gate, qi):
                st = s.reshape([2]*n)
                if gate.shape[0] == 2:
                    psi   = mx.tensordot(gate, st, axes=[[1], [qi]])
                    order = list(range(1, qi+1)) + [0] + list(range(qi+1, n))
                else:
                    g     = gate.reshape(2, 2, 2, 2)
                    psi   = mx.tensordot(g, st, axes=[[2, 3], [qi, qi+1]])
                    order = list(range(2, qi+2)) + [0, 1] + list(range(qi+2, n))
                return mx.transpose(psi, order).reshape([2**n])

            s = apply_gate_flat(s, H, 0)
            for i in range(n - 1):
                s = apply_gate_flat(s, CNOT, i)
            mx.eval(s)

        else:   # direct_index
            s   = mx.zeros([2**n], dtype=mx.complex64)
            s[0] = 1.0
            idx = mx.arange(2**n, dtype=mx.int32)
            mx.eval(s, idx)

            inv_sqrt2 = float(_np.float32(1.0 / _np.sqrt(2)))
            partner   = idx ^ 1
            is_zero   = (idx & 1) == 0
            s = mx.where(is_zero, inv_sqrt2*(s + s[partner]), inv_sqrt2*(s[partner] - s))
            mx.eval(s)

            for i in range(n - 1):
                partner = idx ^ (1 << (i + 1))
                s = mx.where(((idx >> i) & 1) == 1, s[partner], s)
                mx.eval(s)

        return _np.array(s.flatten()[:4])


# ── MLX QFT subprocess runner ─────────────────────────────────────────────────
class _MLXQftRunner:
    def __init__(self, device, variant):
        self.device  = device
        self.variant = variant   # 'tensor', 'flat', or 'direct_index'

    def prepare(self):
        import mlx.core as mx
        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)
        mx.eval(mx.zeros(1, dtype=mx.complex64))

    def __call__(self, n):
        import mlx.core as mx
        import numpy as _np
        import math as _math

        dev = mx.gpu if self.device == 'gpu' else mx.cpu
        mx.set_default_device(dev)

        H_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
        SWAP_np = _np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=_np.complex64)

        def cp_mat(theta):
            return _np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                               [0,0,0,_np.exp(1j*_np.float64(theta))]], dtype=_np.complex64)

        if self.variant == 'tensor':
            s    = mx.zeros([2]*n, dtype=mx.complex64)
            s[tuple([0]*n)] = 1.0
            H    = mx.array(H_np)
            SWAP = mx.array(SWAP_np)

            def apply_1q(s, gate, q):
                nq    = len(s.shape)
                psi   = mx.tensordot(gate, s, axes=[[1], [q]])
                order = list(range(1, q+1)) + [0] + list(range(q+1, nq))
                return mx.transpose(psi, order)

            def apply_2q(s, gate4, q1, q2):
                nq    = len(s.shape)
                g     = gate4.reshape(2, 2, 2, 2)
                psi   = mx.tensordot(g, s, axes=[[2, 3], [q1, q2]])
                order = (list(range(2, q1+2)) + [0] +
                         list(range(q1+2, q2+1)) + [1] +
                         list(range(q2+1, nq)))
                return mx.transpose(psi, order)

            for j in range(n):
                s = apply_1q(s, H, j)
                for k in range(j+1, n):
                    theta = 2.0 * _math.pi / (2 ** (k - j + 1))
                    s = apply_2q(s, mx.array(cp_mat(theta)), j, k)
                mx.eval(s)
            for i in range(n // 2):
                s = apply_2q(s, SWAP, i, n-1-i)
            mx.eval(s)

        elif self.variant == 'flat':
            s    = mx.zeros(2**n, dtype=mx.complex64)
            s[0] = 1.0
            H    = mx.array(H_np)
            SWAP = mx.array(SWAP_np)

            def apply_1q_flat(s, gate, q):
                st    = s.reshape([2]*n)
                psi   = mx.tensordot(gate, st, axes=[[1], [q]])
                order = list(range(1, q+1)) + [0] + list(range(q+1, n))
                return mx.transpose(psi, order).reshape([2**n])

            def apply_2q_flat(s, gate4, q1, q2):
                st    = s.reshape([2]*n)
                g     = gate4.reshape(2, 2, 2, 2)
                psi   = mx.tensordot(g, st, axes=[[2, 3], [q1, q2]])
                order = (list(range(2, q1+2)) + [0] +
                         list(range(q1+2, q2+1)) + [1] +
                         list(range(q2+1, n)))
                return mx.transpose(psi, order).reshape([2**n])

            for j in range(n):
                s = apply_1q_flat(s, H, j)
                for k in range(j+1, n):
                    theta = 2.0 * _math.pi / (2 ** (k - j + 1))
                    s = apply_2q_flat(s, mx.array(cp_mat(theta)), j, k)
                mx.eval(s)
            for i in range(n // 2):
                s = apply_2q_flat(s, SWAP, i, n-1-i)
            mx.eval(s)

        else:   # direct_index
            s   = mx.zeros([2**n], dtype=mx.complex64)
            s[0] = 1.0
            idx = mx.arange(2**n, dtype=mx.int32)
            mx.eval(s, idx)

            def direct_index_h(s, tgt, idx):
                inv_sqrt2 = float(_np.float32(1.0 / _np.sqrt(2)))
                partner   = idx ^ (1 << tgt)
                is_zero   = ((idx >> tgt) & 1) == 0
                return mx.where(is_zero, inv_sqrt2*(s + s[partner]), inv_sqrt2*(s[partner] - s))

            def direct_index_cp(s, q1, q2, theta, idx):
                mask  = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 1)
                phase = complex(_np.exp(1j * _np.float64(theta)))
                return mx.where(mask, s * phase, s)

            def direct_index_swap(s, q1, q2, idx):
                differs = ((idx >> q1) & 1) != ((idx >> q2) & 1)
                partner = idx ^ (1 << q1) ^ (1 << q2)
                return mx.where(differs, s[partner], s)

            for j in range(n):
                s = direct_index_h(s, j, idx);  mx.eval(s)
                for k in range(j+1, n):
                    theta = 2.0 * _math.pi / (2 ** (k - j + 1))
                    s = direct_index_cp(s, j, k, theta, idx);  mx.eval(s)
            for i in range(n // 2):
                s = direct_index_swap(s, i, n-1-i, idx);  mx.eval(s)

        return _np.array(s.flatten()[:4])


# ── JAX implementations (in-process; no Metal crash risk) ─────────────────────
import jax
import jax.numpy as jnp

jax.config.update('jax_platform_name', 'cpu')

_h_mat  = jnp.array([[1,1],[1,-1]], dtype=jnp.complex64) / jnp.sqrt(2)
_cnot_m = jnp.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=jnp.complex64)
_swap_m = jnp.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=jnp.complex64)


@functools.partial(jax.jit, static_argnums=(2, 3))
def _apply_gate_jax(state, gate, qubit, n_q):
    n_gate  = int(np.log2(gate.shape[0]))
    state_t = state.reshape([2] * n_q)
    result  = jnp.tensordot(
        gate.reshape([2] * (n_gate * 2)), state_t,
        axes=([n_gate + k for k in range(n_gate)],
              [qubit + k  for k in range(n_gate)]))
    axes = list(range(n_gate, n_q))
    axes = axes[:qubit] + list(range(n_gate)) + axes[qubit:]
    return jnp.transpose(result, axes).reshape(2 ** n_q)


def ghz_jax(n):
    s = jnp.zeros(2**n, dtype=jnp.complex64).at[0].set(1.0)
    s = _apply_gate_jax(s, _h_mat, 0, n)
    for i in range(n - 1):
        s = _apply_gate_jax(s, _cnot_m, i, n)
    return np.array(s)   # forces XLA completion


def _apply_1q_jax(s, gate, q):
    psi = jnp.tensordot(jnp.array(gate), s, axes=[[1], [q]])
    return jnp.moveaxis(psi, 0, q)


def _apply_2q_jax(s, gate4, q1, q2):
    g   = jnp.array(gate4).reshape(2, 2, 2, 2)
    psi = jnp.tensordot(g, s, axes=[[2, 3], [q1, q2]])
    return jnp.moveaxis(psi, [0, 1], [q1, q2])


def qft_jax(n):
    """Matches bench_qft.py backend C exactly (no explicit JIT; XLA-compiled per call)."""
    s    = jnp.zeros([2]*n, dtype=jnp.complex64).at[tuple([0]*n)].set(1.0)
    H_j  = jnp.array(np.array([[1,1],[1,-1]], dtype=np.complex64) / np.sqrt(2))
    SW_j = jnp.array(np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=np.complex64))
    for j in range(n):
        s = _apply_1q_jax(s, H_j, j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            cp = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                           [0,0,0,np.exp(1j * float(theta))]], dtype=np.complex64)
            s = _apply_2q_jax(s, cp, j, k)
    for i in range(n // 2):
        s = _apply_2q_jax(s, SW_j, i, n-1-i)
    s.block_until_ready()
    return np.array(s)


# ── Thermal helpers ───────────────────────────────────────────────────────────
def _read_thermal_pressure():
    """Returns thermal pressure string ('Nominal', 'Moderate', 'Heavy', 'Critical')
    or None if powermetrics is unavailable / sudo not configured."""
    try:
        r = subprocess.run(
            ['sudo', 'powermetrics', '--samplers', 'thermal', '-i', '500', '-n', '1'],
            capture_output=True, text=True, timeout=20)
        for line in r.stdout.splitlines():
            if 'current pressure level' in line.lower():
                return line.split(':')[1].strip()
    except Exception:
        pass
    return None


def _wait_for_cool(next_backend_name):
    log(f"\n  [Thermal recovery: sleeping {COOL_SECS}s before '{next_backend_name}']")
    time.sleep(COOL_SECS)
    for _ in range(10):
        pressure = _read_thermal_pressure()
        if pressure is None:
            print("\n" + "!"*68, flush=True)
            print("  FATAL: Cannot read thermal pressure via powermetrics.", flush=True)
            print("  Thermal isolation requires confirmed cooling between backends.", flush=True)
            print("  Without it, results will show thermal artifacts that differ", flush=True)
            print("  from the published paper data.", flush=True)
            print("", flush=True)
            print("  To fix, add this line via  sudo visudo  (or tee to sudoers.d):", flush=True)
            print("    ALL ALL=(root) NOPASSWD: /usr/bin/powermetrics", flush=True)
            print("", flush=True)
            print("  Then re-run.  Aborting now to prevent invalid data.", flush=True)
            print("!"*68 + "\n", flush=True)
            sys.exit(1)
        log(f"  Thermal pressure: {pressure}  (target: Nominal)")
        if pressure == 'Nominal':
            log(f"  ✓ Thermal OK — starting next backend")
            return
        log(f"  Still warm ({pressure}) — sleeping 60s more...")
        _flush_log()
        time.sleep(60)
    log(f"  [WARN] Thermal pressure never reached Nominal after 10 extra sleeps — proceeding anyway")


# ── Effective bandwidth ───────────────────────────────────────────────────────
def _eff_bw_ghz(n, t):
    sv_gb = 2**n * 8 / 1e9
    return n * 2 * sv_gb / t           # n gates × 2 passes × sv_gb / s

def _eff_bw_qft(n, t):
    n_h, n_cp, n_sw = n, n*(n-1)//2, n//2
    total = (n_h  * 2**n     * 8 * 2 +
             n_cp * 2**(n-2) * 8 * 2 +
             n_sw * 2**(n-2) * 8 * 4)
    return total / t / 1e9


# ── Logging ───────────────────────────────────────────────────────────────────
_log_lines = []
_log_path  = None

def log(msg=''):
    print(msg, flush=True)
    _log_lines.append(msg)

def _flush_log():
    if _log_path:
        Path(_log_path).write_text('\n'.join(_log_lines))


# ── Trial runner ──────────────────────────────────────────────────────────────
def _run_backend(circuit, bk, fn, isolated, n_trials, timeout_s):
    """1 warmup + n_trials timed runs for each qubit count.
    Returns list of result dicts, one per qubit count.
    """
    eff_bw_fn = _eff_bw_ghz if circuit == 'ghz' else _eff_bw_qft

    log(f"\n{'='*68}")
    log(f"  {circuit.upper()} | {bk}: {BACKENDS[bk]}  |  {n_trials} trials + 1 warmup")
    log(f"{'='*68}")
    log(f"{'q':>4}  {'warmup':>9}  " +
        '  '.join(f'trial{i+1:>2}' for i in range(n_trials)) +
        f"  {'mean':>9}  {'std':>8}  {'cov':>6}  {'ratio':>7}  {'eff_bw':>8}")

    results   = []
    prev_mean = None

    for n in QUBITS:
        sv_gb = 2**n * 8 / 1e9

        # Warmup (discarded)
        t0 = time.perf_counter()
        if isolated:
            _run_isolated(fn, n, timeout_s)
        else:
            fn(n)
        warmup_t = time.perf_counter() - t0

        # Timed trials
        times = []
        for _ in range(n_trials):
            if isolated:
                t = _run_isolated(fn, n, timeout_s)
                if t is None:
                    log(f"  [WARN] Subprocess failed at {n}q — stopping backend {bk}")
                    break
                times.append(t)
            else:
                t0 = time.perf_counter()
                fn(n)
                times.append(time.perf_counter() - t0)

        if not times:
            break

        mean = statistics.mean(times)
        std  = statistics.stdev(times) if len(times) > 1 else 0.0
        cov  = std / mean * 100
        bw   = eff_bw_fn(n, mean)

        ratio_str = f"{mean/prev_mean:>6.2f}×" if prev_mean else "      —"
        cov_flag  = "  *** CoV>5%" if cov > 5.0 else ""
        trial_str = '  '.join(f'{t:>8.3f}' for t in times)

        log(f"{n:>4}  {warmup_t:>9.2f}  {trial_str}  {mean:>9.3f}  "
            f"{std:>8.4f}  {cov:>5.1f}%  {ratio_str}  {bw:>7.2f}"
            f"  [{sv_gb:.2f}GB]{cov_flag}")

        if prev_mean and mean < prev_mean:
            log(f"  *** NOTE: {n}q ({mean:.3f}s) < {n-1}q ({prev_mean:.3f}s) "
                f"— unexpected inverse scaling ***")

        results.append({
            'qubits': n, 'state_gb': sv_gb, 'warmup_s': warmup_t,
            'times': times, 'mean': mean, 'std': std, 'cov': cov,
            'ratio': mean / prev_mean if prev_mean else None,
            'eff_bw': bw,
        })
        prev_mean = mean
        _flush_log()

    # Intra-trial variance
    if n_trials >= 3:
        log(f"\n── Intra-trial variance ({bk}) ──")
        for r in results:
            t  = r['times']
            h1 = statistics.mean(t[:len(t)//2])
            h2 = statistics.mean(t[len(t)//2:])
            trend = ("warming↑" if h2 > h1 * 1.05 else
                     "cooling↓" if h2 < h1 * 0.95 else "stable")
            log(f"  {r['qubits']:2d}q  first={h1:.3f}s  second={h2:.3f}s  → {trend}")

    return results


# ── CSV output ────────────────────────────────────────────────────────────────
def _write_csv(csv_path, circuit, all_results, n_trials):
    t_cols = [f't{i+1}' for i in range(n_trials)]
    fields = (['circuit', 'backend', 'backend_name', 'qubits', 'state_gb', 'warmup_s']
              + t_cols
              + ['mean_s', 'std_s', 'cov_pct', 'ratio', 'eff_bw_gb_s', 'cov_warn'])
    with open(csv_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for bk, rows in all_results.items():
            for r in rows:
                row = {
                    'circuit':      circuit.upper(),
                    'backend':      bk,
                    'backend_name': BACKENDS[bk],
                    'qubits':       r['qubits'],
                    'state_gb':     f"{r['state_gb']:.4f}",
                    'warmup_s':     f"{r['warmup_s']:.3f}",
                    'mean_s':       f"{r['mean']:.4f}",
                    'std_s':        f"{r['std']:.6f}",
                    'cov_pct':      f"{r['cov']:.2f}",
                    'ratio':        f"{r['ratio']:.4f}" if r['ratio'] else '',
                    'eff_bw_gb_s':  f"{r['eff_bw']:.3f}",
                    'cov_warn':     'WARN_COV>5pct' if r['cov'] > 5.0 else '',
                }
                for i, t in enumerate(r['times']):
                    row[f't{i+1}'] = f"{t:.6f}"
                for i in range(len(r['times']), n_trials):
                    row[f't{i+1}'] = ''
                w.writerow(row)
    log(f"CSV: {csv_path}")


# ── Summary ───────────────────────────────────────────────────────────────────
def _print_summary(circuit, all_results):
    qft_note = "  [cross-validation, N=3]" if circuit == 'qft' else ""
    log("\n\n" + "="*68)
    log(f"  SUMMARY — {circuit.upper()} thermally isolated (27–30q){qft_note}")
    log("="*68)

    for bk, rows in all_results.items():
        label = f"{circuit.upper()} {BACKENDS[bk]} (thermally isolated)"
        if circuit == 'qft':
            label += "  [cross-validation, N=3]"
        log(f"\n{label}:")
        log(f"  {'q':>4}  {'mean(s)':>10}  {'std':>8}  {'ratio':>8}  state_vec")
        for r in rows:
            ratio_s  = f"{r['ratio']:.3f}×" if r['ratio'] else "—"
            cov_flag = "  *** CoV>5%" if r['cov'] > 5.0 else ""
            log(f"  {r['qubits']:>4}  {r['mean']:>10.4f}  {r['std']:>8.4f}"
                f"  {ratio_s:>8}  {r['state_gb']:.2f} GB{cov_flag}")

    log("\n── Cliff ratios (28q → 29q) ──")
    for bk, rows in all_results.items():
        r28 = next((r for r in rows if r['qubits'] == 28), None)
        r29 = next((r for r in rows if r['qubits'] == 29), None)
        r30 = next((r for r in rows if r['qubits'] == 30), None)
        if r28 and r29:
            cliff = r29['mean'] / r28['mean']
            post  = f"  29→30q: {r30['mean']/r29['mean']:.2f}×" if r30 else ""
            log(f"  {bk} {BACKENDS[bk]:<22}: 28→29q cliff = {cliff:.2f}×{post}")


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description='Thermally isolated cliff benchmark (27–30q, 4 backends)')
    parser.add_argument('--circuit', required=True, choices=['ghz', 'qft'],
                        help='Circuit to run. Use separate invocations for ghz and qft.')
    parser.add_argument('--backends', default='C,F,J,K',
                        help='Backend keys to run (default: C,F,J,K)')
    parser.add_argument('--no-cool', action='store_true',
                        help='Skip thermal wait (debugging only)')
    args   = parser.parse_args()
    circuit = args.circuit

    if args.no_cool:
        print("\n" + "!"*68, flush=True)
        print("  WARNING: --no-cool is active.", flush=True)
        print("  Thermal recovery between backends is DISABLED.", flush=True)
        print("  Results WILL NOT match the published paper data and should NOT", flush=True)
        print("  be compared against it.  Use only for development / smoke tests.", flush=True)
        print("!"*68 + "\n", flush=True)

    backend_keys = [k.strip().upper() for k in args.backends.split(',')]
    n_trials  = N_TRIALS_GHZ if circuit == 'ghz' else N_TRIALS_QFT
    timeout_s = TIMEOUT_GHZ  if circuit == 'ghz' else TIMEOUT_QFT

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = Path(__file__).parent.parent / "logs"
    logs_dir.mkdir(exist_ok=True)
    global _log_path
    _log_path = str(logs_dir / f"cliff_isolated_{circuit}_{ts}.log")
    csv_path  = str(Path(__file__).parent.parent / f"cliff_isolated_{circuit}_{ts}.csv")

    log("="*68)
    log(f"  Cliff Benchmark — Thermally Isolated — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log(f"  Circuit:   {circuit.upper()}")
    bk_str = ', '.join(f"{k} ({BACKENDS.get(k, '?')})" for k in backend_keys)
    log(f"  Backends:  {bk_str}")
    log(f"  Qubits:    {QUBITS}")
    log(f"  Trials:    {n_trials} + 1 warmup per qubit count")
    cool_str = f"disabled (--no-cool)" if args.no_cool else f"{COOL_SECS}s + thermal pressure == Nominal"
    log(f"  Cooling:   {cool_str}")
    log(f"  Log:       {_log_path}")
    log(f"  CSV:       {csv_path}")
    log("="*68)
    log()
    log("  IMPORTANT: Run under caffeinate to prevent sleep:")
    log(f"    caffeinate -i python3 bench_cliff_isolated.py --circuit {circuit}")
    if circuit == 'qft':
        log()
        log("  QFT results will be labeled 'cross-validation, N=3'.")
        log("  JAX QFT is in-process (no subprocess timeout). 30q ≈ 1000s per trial.")
    log()
    sv_str = '  '.join(f"{n}q={2**n*8/1e9:.2f}GB" for n in QUBITS)
    log(f"  State vectors: {sv_str}")
    log()
    _flush_log()

    # Backend runner map: key → (fn, is_isolated)
    ghz_runners = {
        'C': (ghz_jax,                          False),
        'F': (_MLXGhzRunner('gpu', 'tensor'),   True),
        'G': (_MLXGhzRunner('cpu', 'tensor'),   True),
        'H': (_MLXGhzRunner('gpu', 'flat'),     True),
        'I': (_MLXGhzRunner('cpu', 'flat'),     True),
        'J': (_MLXGhzRunner('gpu', 'direct_index'),     True),
        'K': (_MLXGhzRunner('cpu', 'direct_index'),     True),
    }
    qft_runners = {
        'C': (qft_jax,                           False),
        'F': (_MLXQftRunner('gpu', 'tensor'),    True),
        'G': (_MLXQftRunner('cpu', 'tensor'),    True),
        'H': (_MLXQftRunner('gpu', 'flat'),      True),
        'I': (_MLXQftRunner('cpu', 'flat'),      True),
        'J': (_MLXQftRunner('gpu', 'direct_index'),      True),
        'K': (_MLXQftRunner('cpu', 'direct_index'),      True),
    }
    runners = ghz_runners if circuit == 'ghz' else qft_runners

    all_results = {}
    for i, bk in enumerate(backend_keys):
        if bk not in runners:
            log(f"\n[WARN] Unknown backend '{bk}' — valid keys: {list(runners.keys())}")
            continue
        if i > 0 and not args.no_cool:
            _wait_for_cool(BACKENDS[bk])
        fn, isolated = runners[bk]
        rows = _run_backend(circuit, bk, fn, isolated, n_trials, timeout_s)
        all_results[bk] = rows
        _flush_log()

    _print_summary(circuit, all_results)
    _write_csv(csv_path, circuit, all_results, n_trials)
    log(f"\nLog: {_log_path}")
    _flush_log()


if __name__ == '__main__':
    mp.freeze_support()
    main()
