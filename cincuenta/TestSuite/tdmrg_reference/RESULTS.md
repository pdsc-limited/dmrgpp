# Initial prescribed-bath experiment results

These are historical frozen-window implementation pilots, not the normal
advancing protocol or a converged neq-DMFT calculation. The subsequent
[normal-schedule study](NORMAL_SCHEDULE.md) restores N-2 advancement and the
engine's three-target default and shows much smaller static-bath errors. The
plateau below is not evidence of an intrinsic engine accuracy floor. Engine
source was unchanged (baseline revision `196d86e5`). Release build: Clang 21.1.4,
CPU batched GEMM, active scientific-workstation Spack environment. Python used
NumPy 2.5.3, SciPy 1.18.1, h5py 3.16.0; BLAS/OpenMP threads were set to one.

Protocol: U=2, dt=0.2, three intervals, four insertion times, two pure-state
seeds averaged equally, prescribed changing complex star couplings, one
out-and-back sweep per stage. Total N includes the impurity. Every completed
case executes 47 processes. See README for signs, restart mappings, and limits
of each diagnostic.

## Accuracy and internal consistency

Maximum absolute **complex** GF entry error over both seeds and both channels:

| N / bath sites | m | max GF error vs ED | max paired-reference/master observable spread | min eig(-iG<) | min eig(iG>) |
|---|---:|---:|---:|---:|---:|
| 5 / 4 | 64 | 6.53e-15 | 0 | 1.06e-5 | 8.35e-5 |
| 7 / 6 | 8 | 6.77e-3 | 9.65e-3 | -4.10e-5 | -1.27e-4 |
| 7 / 6 | 32 | 1.76e-4 | 1.37e-4 | 3.98e-5 | 6.56e-5 |
| 7 / 6 | 128 | 1.75e-4 | 1.36e-4 | 4.03e-5 | 6.53e-5 |

Reference observables are impurity nup, ndown, and double occupancy. Their
maximum ED errors across both seeds are:

| N,m | reference-only master | references evolved alongside inserted branches |
|---|---:|---:|
| 5,64 | 3.93e-15 | 3.93e-15 |
| 7,8 | 6.80e-3 | 9.66e-3 |
| 7,32 | 1.43e-4 | 1.63e-4 |
| 7,128 | 1.43e-4 | 1.45e-4 |

The equal-time anticommutator and lesser-diagonal/master-density checks were
zero to printed precision in the one-sweep ladder. The kernel signs above are
for the reconstructed **ensemble** matrices. Small negative eigenvalues at
m=8 were retained, not clipped. Increasing m helps substantially, but positive
kernels at m=32/128 do not imply convergence.

The maximum discarded weights were 0, 2.92e-4, 2.13e-8, and 4.31e-16 respectively;
WFT warning counts were 0, 247, 0, 0. Crucially, the N7 error plateaus even when
the reported discarded weight is nearly zero. A three-sweep-pair N7/m128 run
still had max GF error 1.72e-4 and reference spread 1.24e-4. Extra sweeps did
**not** resolve it. Time-target/projection schedule, propagation accuracy, and
cached block-Hamiltonian refresh need further isolation; none is established
as the cause. These N7 results must not be called untruncated/converged merely
from the discarded-weight diagnostic.

## Continuation and correction/replay

* N5 segmented versus single-window complex amplitude difference: 0 at output
  precision. N7 differences at m=8/32/128: 9.97e-5 / 1.13e-4 / 8.85e-5.
  Different time-target sets make this a combined schedule/restart comparison.
* Every completed case preserved the accepted checkpoint hash and replayed the
  corrected trial amplitude exactly at output precision. Rejected and corrected
  couplings give distinct amplitudes (N5 difference 2.55e-2).
* This checks immutable branching and one observable, not full-state equality,
  an adaptive error controller, or DMFT feedback.

## Process memory, checkpoints, runtime

These are serial child-process RSS measurements, excluding the Python ED
coordinator, from the isolated stdlib wait4 wrapper. Earlier exploratory RSS
numbers contaminated by the scientific parent's inherited RSS are superseded.
MiB means 2^20 bytes. Wall time sums only stage processes, not ED computation.

| N,m | peak child RSS MiB | largest checkpoint MiB | all retained checkpoints MiB | 47-stage wall seconds | segmented minus single-window seconds |
|---|---:|---:|---:|---:|---:|
| 5,64 | 24.7 | 1.45 | 65.4 | 11.30 | 0.237 |
| 7,8 | 25.1 | 1.18 | 53.0 | 12.59 | 0.293 |
| 7,32 | 30.2 | 1.69 | 75.9 | 16.27 | 0.345 |
| 7,128 | 55.5 | 5.90 | 238.0 | 27.16 | 0.503 |

The last column includes extra launch/load/write and differing sweep work; it
is not an isolated restart-load cost. These pilots support the practicality of
small local target sets plus redundant runs at these sizes, not asymptotic or
MPI rank-scaling claims. Retaining every checkpoint is an audit choice.

## Larger-bath blocker and operator-expression workaround

N9/m16 reached multiple trajectory/branch stages but failed at
`alpha_particle_insert_1`: the engine process exited with SIGSEGV. Repeating the
same input without CLI measurements instead aborted with
`An important vector has norm=0 which is too small`, after an absent-partition
message. The exact cause is unresolved. N9 is **not validated**, and the full
larger-bath ladder remains blocked; there is deliberately no dense N9 ED oracle.

Separately, direct `c*Pket` insertion was independently traced to an existing
source-ket overwrite in the targeting expression evaluator. The driver uses
the existing restart-to-gs facility as a workaround, tested at N5/N7. This is
not a general repair of that evaluator. Neither finding led to engine changes.

## Evidence and validation

Local generated evidence (not tracked in Git):

* `build/tdmrg_validated/{results.json,n*/summary.json}`: final four-case run,
  complete complex ED/DMRG matrices, reference ED comparisons, process metrics.
* `build/tdmrg_sweep_check/n7_m128/summary.json`: three-sweep comparison.
* `build/tdmrg_experiments_ladder/n9_m16/alpha_particle_insert_1/`: failed
  larger-bath input, log, job metadata, and partial checkpoint.
* `build/tdmrg_n9_repro/`: repeated insertion without measurement arguments.
* `build/tdmrg-final-ctest.log`: 12 focused engine CTests passed again.
* `build/tdmrg-unit-tests.log`: all 17 Python unit tests passed.

Unit tests cover analytic atomic limits, independent full-Fock nonstationary
complex correlators (including active interactions and both spin probes),
1-RDM invariants, input/mapping generation, log parsing, process failures,
timeouts, and child-RSS isolation from a 160 MiB Python parent. Independent
review found no confirmed numerical harness defect and requested the stronger
interacting/spin-down oracle checks. Future work should resolve the N7 plateau
and N9 insertion failure before extending bath sizes or adding DMFT feedback.
