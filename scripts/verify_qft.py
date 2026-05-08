"""
verify_qft.py

Correctness check for all QFT backends.
QFT|00..0> = uniform superposition: all 2^n amplitudes = 1/sqrt(2^n), real, positive.

Usage:
  python3 scripts/verify_qft.py
"""
import os, math
import numpy as np

os.environ['JAX_PLATFORMS'] = 'cpu'

N   = 5      # qubit count
TOL = 1e-5


def _expected(n):
    return np.full(2**n, 1.0 / math.sqrt(2**n), dtype=np.complex64)


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
        bad = np.where(np.abs(s - exp) > TOL)[0][:6]
        print(f"         wrong indices (first 6): {bad.tolist()}")
        print(f"         got:      {s[bad].tolist()}")
        print(f"         expected: {exp[bad].tolist()}")
    return ok


# ── JAX CPU tensordot (C) ─────────────────────────────────────────────────────
def _run_jax(n):
    import jax.numpy as jnp
    import numpy as _np

    H_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
    SWAP_np = _np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=_np.complex64)

    def cp_mat(theta):
        return _np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                           [0,0,0,_np.exp(1j*_np.float64(theta))]], dtype=_np.complex64)

    def apply_1q(s, gate, q):
        psi = jnp.tensordot(jnp.array(gate), s, axes=[[1], [q]])
        return jnp.moveaxis(psi, 0, q)

    def apply_2q(s, gate4, q1, q2):
        g   = jnp.array(gate4).reshape(2, 2, 2, 2)
        psi = jnp.tensordot(g, s, axes=[[2, 3], [q1, q2]])
        return jnp.moveaxis(psi, [0, 1], [q1, q2])

    H_j  = jnp.array(H_np)
    SW_j = jnp.array(SWAP_np)
    s    = jnp.zeros([2]*n, dtype=jnp.complex64).at[tuple([0]*n)].set(1.0)

    for j in range(n):
        s = apply_1q(s, H_j, j)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = apply_2q(s, cp_mat(theta), j, k)
    for i in range(n // 2):
        s = apply_2q(s, SW_j, i, n-1-i)
    s.block_until_ready()
    return np.array(s).flatten()


# ── MLX tensordot (F/G) ───────────────────────────────────────────────────────
def _run_mlx_tensor(device_str, n):
    import mlx.core as mx
    import numpy as _np

    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    H_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
    SWAP_np = _np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=_np.complex64)

    def cp_mat(theta):
        return _np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                           [0,0,0,_np.exp(1j*_np.float64(theta))]], dtype=_np.complex64)

    H    = mx.array(H_np)
    SWAP = mx.array(SWAP_np)
    s    = mx.zeros([2]*n, dtype=mx.complex64)
    s[tuple([0]*n)] = 1.0

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
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = apply_2q(s, mx.array(cp_mat(theta)), j, k)
        mx.eval(s)
    for i in range(n // 2):
        s = apply_2q(s, SWAP, i, n-1-i)
    mx.eval(s)
    return np.array(s.flatten())


# ── MLX flat-index (H/I) ──────────────────────────────────────────────────────
def _run_mlx_flat(device_str, n):
    import mlx.core as mx
    import numpy as _np

    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    H_np    = _np.array([[1,1],[1,-1]], dtype=_np.complex64) / _np.sqrt(2)
    SWAP_np = _np.array([[1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]], dtype=_np.complex64)

    def cp_mat(theta):
        return _np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],
                           [0,0,0,_np.exp(1j*_np.float64(theta))]], dtype=_np.complex64)

    H    = mx.array(H_np)
    SWAP = mx.array(SWAP_np)
    s    = mx.zeros(2**n, dtype=mx.complex64)
    s[0] = 1.0

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
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = apply_2q_flat(s, mx.array(cp_mat(theta)), j, k)
        mx.eval(s)
    for i in range(n // 2):
        s = apply_2q_flat(s, SWAP, i, n-1-i)
    mx.eval(s)
    return np.array(s.flatten())


# ── MLX direct-index (J/K) ────────────────────────────────────────────────────
def _run_mlx_direct(device_str, n):
    import mlx.core as mx
    import numpy as _np

    dev = mx.gpu if device_str == 'gpu' else mx.cpu
    mx.set_default_device(dev)

    s   = mx.zeros([2**n], dtype=mx.complex64)
    s[0] = 1.0
    idx = mx.arange(2**n, dtype=mx.int32)
    mx.eval(s, idx)

    inv_sqrt2 = float(_np.float32(1.0 / _np.sqrt(2)))

    def direct_h(s, tgt):
        partner = idx ^ (1 << tgt)
        is_zero = ((idx >> tgt) & 1) == 0
        return mx.where(is_zero, inv_sqrt2*(s + s[partner]),
                                  inv_sqrt2*(s[partner] - s))

    def direct_cp(s, q1, q2, theta):
        mask  = (((idx >> q1) & 1) == 1) & (((idx >> q2) & 1) == 1)
        phase = complex(_np.exp(1j * _np.float64(theta)))
        return mx.where(mask, s * phase, s)

    def direct_swap(s, q1, q2):
        differs = ((idx >> q1) & 1) != ((idx >> q2) & 1)
        partner = idx ^ (1 << q1) ^ (1 << q2)
        return mx.where(differs, s[partner], s)

    for j in range(n):
        s = direct_h(s, j);  mx.eval(s)
        for k in range(j+1, n):
            theta = 2.0 * math.pi / (2 ** (k - j + 1))
            s = direct_cp(s, j, k, theta);  mx.eval(s)
    for i in range(n // 2):
        s = direct_swap(s, i, n-1-i);  mx.eval(s)

    return np.array(s.flatten())


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print(f"\nQFT correctness check  (n={N}, tol={TOL:.0e})")
    print(f"Expected: all {2**N} amplitudes = 1/sqrt({2**N}) = {1/math.sqrt(2**N):.6f}")
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
        print("All backends produce correct QFT state.")
