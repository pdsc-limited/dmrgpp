# Wolf et al. ITensors.jl reference

A standalone Julia/ITensors.jl prototype for the nonequilibrium SIAM
benchmark in Wolf, McCulloch, and Schollwoeck, *Solving nonequilibrium DMFT
using matrix product states*, arXiv:1410.3342, Sec. "Results for a
non-self-consistent impurity problem" (source: `../dmrgpp_papers/solving neq-dmft using mps/arxiv_source/neqdmft.tex`, lines 1363--1411).

It is deliberately independent of cincuenta and dmrg.  It is a readable
reference and a small-system test bed, not a production solver.

## Benchmark fixed by the paper

The impurity hybridization is

```
Lambda(t,t') = v(t) g(t,t') v(t')
g^{>/<}(t,t') = -/+ i integral dω f^{>/<}(ω) A(ω) exp(-i ω (t-t'))
A(ω) = sqrt(4 - ω^2)/(2π),  T = 1.
```

The hopping ramp starts in the atomic limit and ends at `v0 = 1`:

```
v(t) = (1 - cos(π t / t1))/2   (t < t1)
v(t) = 1                        (t >= t1)
```

The reported observable is the impurity double occupancy
`d(t) = <n_up n_down>` for `U/v0 = 4` and `10`.  The paper reports agreement
of the `Lb = 20` and `22` bath discretizations through approximately
`t = 11/v0`.

## Staged implementation

1. Construct the paper-style finite star bath on a real-time grid: factor
   `-i Lambda^<` and `i Lambda^>` into equal-rank occupied and empty sectors,
   and validate their reconstruction against the continuum kernels above.
   This product-state bath is a dilation of the stated `T=1` hybridization,
   not a zero-temperature surrogate.
2. Resolve and implement the complete initial-state protocol, time-dependent
   star Hamiltonian, and Wolf-style global MPS Krylov propagator. Local TDVP
   and TEBD are deferred. Do not run an MPS calculation with fewer than `Lb=8`,
   and do not make `Lb<8` an application/regression input at all.
   These bath sizes are traps: the paper publishes no results below `Lb=10`,
   so they cannot be treated as useful physics or implementation targets.
3. Make the first physics comparison at bath sizes large enough to expose the
   published convergence trend: start around `Lb=10--16` (the earlier ED
   work cited by Wolf reached `Lb=16`), then target Wolf's `Lb=20,22` result
   through `t≈11`.  We will not claim bath-size convergence merely from
   agreement with ED at a matched finite bath; Wolf et al. do not provide
   such a comparison.
4. Only after that, consider the separate self-consistent NEQ-DMFT problem.

The paper specifies the finite-temperature continuum hybridization but the
supplied source does not give a numerical ramp duration `t1`; its
non-self-consistent section also does not unambiguously settle the impurity
initial-state protocol. These are explicit blockers, not parameters to infer.
See [`docs/benchmark_contract.md`](docs/benchmark_contract.md) and the
baseline configurations in [`configs/`](configs/).

## Environment

Julia is supplied by the `dmrgpp` Spack environment at
`/workspace/spack_env/dmrgpp`; Julia dependencies belong in this directory's
`Project.toml` and generated `Manifest.toml`, not in a global Julia depot.
Once Julia is available, initialize with:

```bash
julia --project=. -e 'using Pkg; Pkg.add(["ITensors", "ITensorMPS"]); Pkg.precompile()'
```

Generated trajectories, plots, and local Julia depot/precompile files must
not be committed. Use this directory's ignored `tmp/` for scratch scripts and
logs; do not use a shared system `/tmp` for this reference.
