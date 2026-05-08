# qsim-uma: Quantum Circuit Simulation on Unified Memory Architecture

Benchmark suite comparing 11 Python backends for quantum state-vector simulation on Apple Silicon (M4 Pro, 48 GB unified memory). Accompanies the paper:

> **"A Controlled Study of Memory Hierarchy Transitions in Quantum State-Vector Simulation on Unified Memory Architecture"**  
> Gyan Pratipat — ASU

---

## Key Findings

- **DRAM cliff at 28→29q**: state vector grows from 2.1 GB to 4.3 GB, exceeding the M4 Pro L3 cache. Tensordot backends show a 3.8–4.5× step; direct-index backends show ~2.1×.
- **Direct-index is consistently DRAM-bound**: scatter-write access pattern prevents cache reuse even at 27q, so there is no cache phase to collapse from — hence the flat ~2× scaling.
- **Cliff location is circuit-independent**: GHZ (O(n) gates) and QFT (O(n²) gates) produce the same cliff magnitude for each backend (±0.1×).
- **JAX XLA/AMX CPU ≈ MLX Metal GPU** for tensordot workloads — both are memory-bandwidth-bound on the same DRAM.
- **30q performance** (fastest → slowest): J (MLX GPU direct-index, 11.3s) → F (MLX GPU tensor, 23.4s) → H (MLX GPU flat, 32.3s) → C (JAX CPU tensordot, 38.3s) → …

---

## Backend Legend

| Key | Backend | Framework | Device | Algorithm |
|-----|---------|-----------|--------|-----------|
| A | Brute-force NumPy | NumPy | CPU | Dense matmul |
| B | pykronecker | pykronecker | CPU | Kronecker product |
| C | JAX CPU tensordot | JAX | CPU (AMX) | tensordot |
| D | NumPy direct-index | NumPy | CPU | Direct-index scatter-write |
| F | MLX GPU tensor | MLX | GPU (Metal) | tensordot |
| G | MLX CPU tensor | MLX | CPU | tensordot |
| H | MLX GPU flat | MLX | GPU (Metal) | Flat-index |
| I | MLX CPU flat | MLX | CPU | Flat-index |
| J | MLX GPU direct-index | MLX | GPU (Metal) | Direct-index scatter-write |
| K | MLX CPU direct-index | MLX | CPU | Direct-index scatter-write |

Backend A terminated before 16q (OOM). Backend E not listed (internal numbering gap).

---

## Requirements

- Apple Silicon Mac (tested: M4 Pro, 48 GB unified memory, macOS 15)
- Python 3.12
- `caffeinate` (built into macOS) to prevent sleep during long runs

```bash
pip install -r requirements.txt
# or: conda env create -f environment.yml
```

Pinned versions used to produce the published results:

```
mlx==0.29.3
jax==0.4.30  jaxlib==0.4.30
numpy==2.0.2
```

> **Thermal monitoring** (required for thermally isolated benchmarks):
> The isolation scripts read thermal pressure via `sudo powermetrics`.
> Add this line to avoid interactive password prompts:
> ```bash
> echo "ALL ALL=(root) NOPASSWD: /usr/bin/powermetrics" | sudo tee /etc/sudoers.d/powermetrics
> ```
> Run `python3 thermal_monitor.py` to verify the setup before benchmarking.

---

## Repository Layout

```
experiments/
  1_ghz_statistical/scripts/  # Exp 1: GHZ, 11 backends, N=7, 3–30q
  2_ghz_isolated/scripts/     # Exp 2: GHZ thermally isolated, N=5, 27–30q (backends C,F,G,H,I,J,K)
  3_qft_single_run/scripts/   # Exp 3: QFT, 11 backends, N=1, 3–30q
  4_qft_isolated/scripts/     # Exp 4: QFT thermally isolated, N=3, 27–30q

scripts/                      # Canonical top-level scripts (mirrors experiment scripts)
  bench_cliff_isolated.py     # Thermally isolated cliff benchmark (27–30q)
  quantum_benchmark.py        # Full 11-backend statistical benchmark (3–30q)
  bench_qft.py                # QFT benchmark
  verify_ghz.py               # Correctness checks
  verify_qft.py
  verify_direct_index.py
  stream_probe.py             # STREAM bandwidth measurement

publication_scripts/          # Regenerate paper figures (data hardcoded in scripts)
  plot_ghz_speedup.py         → fig3
  plot_qft_speedup.py         → fig4
  plot_circuit_independence.py → fig5

figures/                      # Pre-generated publication figures (fig1–fig5)

thermal_monitor.py            # Verify thermal monitoring before benchmarking
requirements.txt
environment.yml
```

---

## Reproducing the Paper Figures

Figures 3–5 can be regenerated directly from the scripts (data is hardcoded):

```bash
python3 publication_scripts/plot_ghz_speedup.py
python3 publication_scripts/plot_qft_speedup.py
python3 publication_scripts/plot_circuit_independence.py
```

Figures 1–2 are included as pre-generated PNGs; their generation scripts depend on raw log files not distributed in this repo.

---

## Reproducing the Benchmark Data

> Run under `caffeinate` to prevent the system from sleeping mid-benchmark.

**Thermally isolated GHZ cliff (~2.5 hours):**
```bash
caffeinate -i python3 scripts/bench_cliff_isolated.py --circuit ghz --backends C,F,J,K
```

**Thermally isolated QFT cliff (~5–6 hours):**
```bash
caffeinate -i python3 scripts/bench_cliff_isolated.py --circuit qft --backends C,F,J,K
```

**Full 11-backend GHZ statistical benchmark (~several hours):**
```bash
caffeinate -i python3 scripts/quantum_benchmark.py
```

> `--no-cool` skips thermal recovery and is for development/smoke-testing only.
> Results from `--no-cool` runs **will not match the paper data**.

---

## Correctness Verification

```bash
python3 scripts/verify_ghz.py
python3 scripts/verify_qft.py
python3 scripts/verify_direct_index.py
```

---

## Citation

```bibtex
@misc{qsimuma2026,
  author       = {Pratipat, Gyan},
  title        = {{qsim-uma}: Quantum Circuit Simulation Benchmarks on Unified Memory Architecture},
  year         = {2026},
  howpublished = {GitHub},
  url          = {https://github.com/gyanpratipat/qsim-uma}
}
```

---

## License

MIT
