"""
verify_ghz.py

Correctness check for all GHZ backends.
Expected final state: (|00..0> + |11..1>) / sqrt(2)
  s[0] = s[2^n - 1] = 1/sqrt(2), all other amplitudes = 0

Usage:
  python3 scripts/verify_ghz.py
"""
import os, math
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'

N   = 5      # qubit count — small enough to be instant, checks all 32 amplitudes
TOL = 1e-5

inv_sqrt2 = float(np.float32(1.0 / np.sqrt(2)))

H_np    = np.array([[1, 1], [1, -1]], dtype=np.complex64) / np.sqrt(2)
CNOT_np = np.array([[1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0]], dtype=np.complex64)


def _expected(n):
    s = np.zeros(2**n, dtype=np.complex64)
    s[0]  = inv_sqrt2
    s[-1] = inv_sqrt2
    return s


def _report(name, state_flat):
    s   = np.array(state_flat).flatten().astype(np.complex64)
    n   = int(round(math.log2(len(s))))
    exp = _expected(n)
    norm    = float(np.linalg.norm(s))
    max_err = float(np.max(np.abs(s - exp)))
    ok  = max_err < TOL and abs(norm - 1.0) < TOL
    tag = "PASS" if ok else "FAIL"
    print(f"  [{tag}] {name:<34s}  norm={norm:.6f}  max_err={max_err:.2e}")
    if not ok:
        bad = np.where(np.abs(s - exp) > TOL)[0]
        print(f"         wrong indices: {bad.tolist()}")
        print(f"         got:      {s[bad].tolist()}")
        print(f"         expected: {exp[bad].tolist()}")
    return ok


# ── JAX CPU tensordot (C) ─────────────────────────────────────────────────────
def _run_jax(n):
    import jax
    import jax.numpy as jnp

    def apply_gate(s, gate, qi, nq):
        ng = int(round(math.log2(gate.shape[0])))
        st = s.reshape([2]*nq)
        r  = jnp.tensordot(gate.reshape([2]*(ng*2)), st,
                           axes=([ng + k for k in range(ng)],
                                 [qi + k  for k in range(ng)]))
        axes = list(range(ng, nq))
        axes = axes[:qi] + list(range(ng)) + axes[qi:]
        return jnp.transpose(r, axes).reshape(2**nq)

    H    = jnp.array(H_np)
    CNOT = jnp.array(CNOT_np)
    s    = jnp.zeros(2**n, dtype=jnp.complex64).at[0].set(1.0)
    s    = apply_gate(s, H, 0, n)
    for i in range(n - 1):
        s = apply_gate(s, CNOT, i, n)
    s.block_until_ready()
    return np.array(s)


# ── MLX tensordot (F/G) ───────────────────────────────────────────────────────
def _run_mlx_tensor(device_str, n):
    import mlx.core as mx
    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    H    = mx.array(H_np)
    CNOT = mx.array(CNOT_np)
    s    = mx.zeros([2]*n, dtype=mx.complex64)
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
    return np.array(s.flatten())


# ── MLX flat-index (H/I) ──────────────────────────────────────────────────────
def _run_mlx_flat(device_str, n):
    import mlx.core as mx
    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    H    = mx.array(H_np)
    CNOT = mx.array(CNOT_np)
    s    = mx.zeros(2**n, dtype=mx.complex64)
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
    return np.array(s.flatten())


# ── MLX direct-index (J/K) ────────────────────────────────────────────────────
def _run_mlx_direct(device_str, n):
    import mlx.core as mx
    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    s   = mx.zeros([2**n], dtype=mx.complex64)
    s[0] = 1.0
    idx = mx.arange(2**n, dtype=mx.int32)
    mx.eval(s, idx)

    # Hadamard on qubit 0
    partner = idx ^ 1
    is_zero = (idx & 1) == 0
    s = mx.where(is_zero, inv_sqrt2 * (s + s[partner]),
                           inv_sqrt2 * (s[partner] - s))
    mx.eval(s)

    # CNOT chain: control=i, target=i+1
    for i in range(n - 1):
        partner = idx ^ (1 << (i + 1))
        s = mx.where(((idx >> i) & 1) == 1, s[partner], s)
        mx.eval(s)

    return np.array(s.flatten())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nGHZ correctness check  (n={N}, tol={TOL:.0e})")
    print("=" * 60)

    results = []

    print("\nJAX:")
    results.append(_report("C — JAX CPU tensordot",    _run_jax(N)))

    print("\nMLX GPU:")
    results.append(_report("F — MLX GPU tensor",       _run_mlx_tensor('gpu', N)))
    results.append(_report("H — MLX GPU flat-index",   _run_mlx_flat('gpu', N)))
    results.append(_report("J — MLX GPU direct-index", _run_mlx_direct('gpu', N)))

    print("\nMLX CPU:")
    results.append(_report("G — MLX CPU tensor",       _run_mlx_tensor('cpu', N)))
    results.append(_report("I — MLX CPU flat-index",   _run_mlx_flat('cpu', N)))
    results.append(_report("K — MLX CPU direct-index", _run_mlx_direct('cpu', N)))

    print("\n" + "=" * 60)
    passed = sum(results)
    total  = len(results)
    print(f"Result: {passed}/{total} backends passed")
    if passed < total:
        print("FAIL — see above for details")
        raise SystemExit(1)
    else:
        print("All backends produce correct GHZ state.")
