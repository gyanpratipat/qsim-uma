import numpy as np
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

def direct_index_h_gate(state, target, n_q):
    inv_sqrt2 = np.float32(1.0 / np.sqrt(2))
    idx = np.arange(len(state), dtype=np.int64)
    bit_pos = target  # little-endian: qubit 0 = LSB, matching Qiskit
    mask0 = ((idx >> bit_pos) & 1) == 0
    mask1 = ~mask0
    a = state[mask0].copy()
    b = state[mask1].copy()
    state[mask0] = inv_sqrt2 * (a + b)
    state[mask1] = inv_sqrt2 * (a - b)
    return state

def direct_index_cnot_gate(state, ctrl, tgt, n_q):
    idx = np.arange(len(state), dtype=np.int64)
    ctrl_bit = ctrl
    tgt_bit  = tgt
    ctrl_mask = ((idx >> ctrl_bit) & 1) == 1
    tgt0_mask = ((idx >> tgt_bit)  & 1) == 0
    swap_mask = ctrl_mask & tgt0_mask
    swap_idx  = idx[swap_mask]
    partner   = swap_idx ^ (1 << tgt_bit)
    tmp = state[swap_idx].copy()
    state[swap_idx] = state[partner]
    state[partner]  = tmp
    return state

def direct_index_ghz(n):
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    state = direct_index_h_gate(state, 0, n)
    for i in range(n-1):
        state = direct_index_cnot_gate(state, i, i+1, n)
    return state

def qiskit_ghz(n):
    qc = QuantumCircuit(n)
    qc.h(0)
    for i in range(n-1):
        qc.cx(i, i+1)
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    result = sim.run(qc).result()
    sv = np.array(result.get_statevector())
    return sv.astype(np.complex64)

# Test across multiple qubit counts and circuit types
print("GHZ Circuit Verification — direct-index vs Qiskit Aer")
print("="*50)

for n in [3, 5, 8, 10, 14]:
    direct_index_sv  = direct_index_ghz(n)
    qiskit_sv = qiskit_ghz(n)

    # Compare amplitudes (global phase may differ)
    match = np.allclose(
        np.abs(direct_index_sv),
        np.abs(qiskit_sv),
        atol=1e-5
    )
    max_err = np.max(np.abs(np.abs(direct_index_sv) - np.abs(qiskit_sv)))
    print(f"  {n:2d}q GHZ: {'PASS' if match else 'FAIL'}  max_err={max_err:.2e}")

# Test arbitrary qubit targeting
print()
print("Arbitrary Target Verification")
print("="*50)

def qiskit_circuit(n, gates):
    """gates = list of ('h', qubit) or ('cx', ctrl, tgt)"""
    qc = QuantumCircuit(n)
    for g in gates:
        if g[0] == 'h':
            qc.h(g[1])
        elif g[0] == 'cx':
            qc.cx(g[1], g[2])
    qc.save_statevector()
    sim = AerSimulator(method='statevector')
    result = sim.run(qc).result()
    return np.array(result.get_statevector(), dtype=np.complex64)

def direct_index_circuit(n, gates):
    state = np.zeros(2**n, dtype=np.complex64)
    state[0] = 1.0
    for g in gates:
        if g[0] == 'h':
            state = direct_index_h_gate(state, g[1], n)
        elif g[0] == 'cx':
            state = direct_index_cnot_gate(state, g[1], g[2], n)
    return state

test_circuits = [
    (5, [('h',2), ('cx',2,4)],            "H on qubit 2, CNOT ctrl=2 tgt=4"),
    (6, [('h',0), ('cx',0,5)],            "H on qubit 0, CNOT ctrl=0 tgt=5 (non-adjacent)"),
    (8, [('h',3), ('cx',3,7), ('cx',3,1)],"H on qubit 3, two CNOTs"),
    (6, [('h',5), ('cx',5,0)],            "H on last qubit, CNOT reversed"),
]

for n, gates, desc in test_circuits:
    direct_index_sv   = direct_index_circuit(n, gates)
    qiskit_sv = qiskit_circuit(n, gates)
    match = np.allclose(np.abs(direct_index_sv), np.abs(qiskit_sv), atol=1e-5)
    max_err = np.max(np.abs(np.abs(direct_index_sv) - np.abs(qiskit_sv)))
    print(f"  {'PASS' if match else 'FAIL'}  {desc}  max_err={max_err:.2e}")
