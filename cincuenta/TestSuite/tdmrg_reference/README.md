# Prescribed-bath traditional t-DMRG experiments

Standalone experimental driver for the existing renormalized-block DMRG++
`TargetingExpression` engine. No MPS/MPO redesign, production-engine edits, or
DMFT feedback. Small baths test implementation correctness, **not** physical
DMFT convergence. Start with the [normal advancement/tau study](NORMAL_SCHEDULE.md)
for the engine's N-2 advancement and three-target defaults. The original driver
below is a frozen-window restart stress test, not the normal advancing protocol.
See [RESULTS.md](RESULTS.md) for its historical outcomes and limitations.

## Running

For the normal advancement control:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \\
  cincuenta/TestSuite/tdmrg_reference/run_normal_schedule.py \\
  --executable build/dmrg/dmrg --output build/normal_new_run \\
  --m 256 --bath-bandwidth 2 --taus .2,.1,.05 --target-times 3
```

Build `dmrg` in the project's activated Spack environment. Python requires
NumPy, SciPy, and h5py; the process/log utilities themselves use only stdlib.
From the source root, for example:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover \
  -s cincuenta/TestSuite/tdmrg_reference -v
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \
  cincuenta/TestSuite/tdmrg_reference/run_experiments.py \
  --executable build/dmrg/dmrg --output build/tdmrg_new_run \
  --cases 5:64,7:8,7:32,7:128 --dt .2 --steps 3 --sweep-pairs 1
```

`N:m` specifies total sites (impurity included) and retained states. The chosen
filled/empty-bath product-state construction requires odd N >= 3. Use a fresh
output directory: existing stage directories are never overwritten. Failures
retain inputs, logs, partial checkpoints, and `job.json`; there is no automated
resume. N > 7 skips the dense ED oracle to bound its memory, **not** to imply
accuracy. The N9/m16 test currently fails in the engine.

## Model and protocol

The impurity has U=2 by default and potential -U/2; other site potentials and
bath energies are zero. Hopping is
`V_b c_b^dagger c_0 + conj(V_b) c_0^dagger c_b` for each spin. Complex couplings
change between frozen intervals; their exact values are saved in the summary.
There is no global U/4 constant. `EnergyForExp=0` avoids an artificial relative
phase when different particle sectors evolve.

Initially half the bath sites are doubly occupied and half empty. Alpha/beta
seeds have respectively up/down impurity occupation and are averaged equally.
Only spin-up Green functions are measured by the driver. The independent oracle
also supports spin-down, with separate-spin bitmasks appropriate to fixed-sector,
same-spin observables (not arbitrary mixed-spin operator products).

For each seed the default suite performs:

* A reference-only trajectory through three bath intervals.
* Lesser/greater insertions at each of four grid times, followed by paired
  branch/reference evolution. Identically zero initial channels are skipped.

Each case additionally uses alpha's initial hole branch for:

* A constant-bath two-interval continuation versus a single evolution window.
* A rejected trial, changed couplings from the immutable accepted checkpoint,
  and an independent fresh-prefix replay.

This is 47 small processes per case. Each has only its local target set; extra
runs and checkpoints deliberately trade work/storage for low process memory.
All checkpoints are retained for auditing, not a minimal-storage production
policy. No simultaneous-target-all-insertions benchmark is claimed.

## Engine details that matter

* `PureState` uses symmetry indices UP=2, DOWN=1, DOUBLE=3, EMPTY=0, not Fock bits.
* Ainur complex numbers use `real i imaginary` without spaces, e.g. `.3i-.1`.
* `TimeEvolve.steps` is the number of target times **within one interval**.
  The normal interval uses two. `advanceEach=1000000` keeps the input fixed
  during the finite sweeps; each stage recomputes one endpoint while adapting
  the basis. `--sweep-pairs` controls full out-and-back sweeps.
* A reference-only interval retains endpoint slot 1; paired intervals retain
  slots 2 and 3. Restart mappings select these slots explicitly. HDF5 endpoint
  times are checked rather than trusting the printed `.last` measurement clock,
  which is the interval's base clock. Measurements use `-p 14`.
* Applying an operator directly to a saved P ket currently has an engine bug:
  `P1="c[0]*|P0>"` can be overwritten by the source P0 after operator application.
  `KetForTargetingExpression::multiply` leaves the old P index; the final
  `TargetingExpression::computePvectors` copy then replaces the result. The
  harness instead uses the existing restart-to-gs facility. This temporary
  backend is isolated in `insertion_restart.plan_insertion`; the physics loop
  consumes its explicit mappings, ket roles, and clock origin. See
  [PVECTOR_INSERTION_BUG.md](PVECTOR_INSERTION_BUG.md) for the focused reproducer
  and engine-fix/removal checklist. It is tested at N5/N7, not a general P-ket
  repair. No HDF5 state editing is used.
* Promoting an endpoint to gs resets its clock. The insertion adapter owns that
  reset and supplies `physical_clock_origin`; the driver validates the declared
  insertion clock rather than assuming all backends reset it.
* Checkpoints retain block Hamiltonians. Arbitrary changed-connector schedules
  must not be assumed to refresh all cached internal terms. These frozen-interval
  sweeps pass N5 ED checks, but the N7 plateau remains unexplained.

## Outputs and interpretation

`results.json` records executable/source hashes, revision, environment versions,
and completed case summaries. Each `summary.json` includes complex matrices as
`[real,imag]`, ED errors and matrices when enabled, reference observables,
endpoint clocks, resources, and stage paths. `job.json` retains raw child stdout,
stderr, DMRG's redirected `runFor*.cout`, parsed weights/norms/discarded weights,
warnings, and checkpoint sizes. A failed case has no completed summary.

Conventions are `G<(t,t')=i<c†(t')c(t)>` and
`G>(t,t')=-i<c(t)c†(t')>`. The upper triangle is reconstructed from
`G(t,t')=-conj(G(t',t))`; anti-Hermiticity is therefore **not an independent
DMRG check**. Full complex causal entries are compared with ED. Equal-time
anticommutators, density consistency, reference-only versus paired-reference
spread, and ensemble-kernel minima of `-iG<` and `iG>` are separate diagnostics.
No positivity clipping is applied. PSD of the reconstructed ensemble kernel
neither proves per-seed positivity nor numerical convergence.

Replay/continuation comparisons use one complex impurity amplitude, not full
wavefunction equality. Accepted checkpoint hashes establish immutability.
Segmented and single-window runs use different targeting schedules (two times
per interval versus three in one window), so truncation/projection differences
are included. Their timing difference includes process launch, load, sweep work,
and checkpoint writing; it is **not an isolated restart-load benchmark**.

Peak process RSS is measured with POSIX `wait4` in a fresh stdlib helper, avoiding
inheritance of the scientific Python parent's large RSS baseline. It excludes
the ED oracle and coordinator. This is a serial process measurement, not an MPI
rank-scaling study. Runtime is the sum of stage wall times, excluding ED. Raw
logs report the engine's virtual memory separately; it is not RSS.
