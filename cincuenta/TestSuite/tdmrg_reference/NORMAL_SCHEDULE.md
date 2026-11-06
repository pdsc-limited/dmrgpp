# Conventional advancement and tau controls

This follow-up separates the normal advancing protocol from the frozen-window,
restart-per-interval experiments in RESULTS.md. It does not add DMFT feedback.

## Energy origin

`TimeEvolve{EnergyForExp=Eref}` uses `exp[-i(H-Eref)t]`, not a Hamiltonian with
energy fixed to Eref. The parser defaults to the targeting auxiliary's current
energy; an explicit value overrides it. In the Krylov implementation the shift
is subtracted from each Ritz eigenvalue in the exponential:

* `dmrg/Engine/NonLocalForTargetingExpression.h`: TimeParams initialization and
  EnergyForExp parsing.
* `dmrg/Engine/TimeVectorsKrylov.h`: calcTargetVector, `(eigsii[k]-Eg)*time`.

An identical shift on both propagated states cancels in their overlap. Different
origins or inconsistent accumulated phases between independently evolved sectors
would not cancel. Setting zero consistently makes the convention explicit; it
does not remove U, potentials, bath energies, or hopping. As a direct control,
repeating both N7 tau=0.1 paired runs with the common origin changed from 0 to
0.37 changed the measured amplitudes by at most 1.56e-11 (alpha) and 5.45e-12
(beta), consistent within the numerical accuracy of these runs. Evidence:
`build/tdmrg_energy_origin_check/comparison.json`.

## Schedule and clock interpretation

The expression default is `advanceEach=nsites-2`. Advancement is also gated to
a boundary; the counter is not simply a physical timestep count. For N7 the
normal setting is 5, not the earlier artificial 1000000 used to prevent any
advancement within a process.

The normal control uses one continuous paired-evolution process, `tau` equal
to the window duration, and a constant Hamiltonian. Both two and three target
times per window were tested; **three is the engine default** and is now the
runner default.
Seed, insertion, and evolution all use the same active bath connectors. This
avoids a zero-hopping seed followed by a changed-Hamiltonian restart. One
out-and-back sweep comprises two `@auto` directional legs. With the tested
restart/border convention, L even legs produce paired checkpoint times
`[L*tau,L*tau,(L+1)*tau,(L+1)*tau]` for two targets. With three targets the
intermediate half-window times are also present: `[L*tau,L*tau,(L+.5)*tau,
(L+1)*tau,(L+.5)*tau,(L+1)*tau]`.

Impurity cocoon measurements occur before and after advancement at site zero.
Their printed clock is the window base; `.last` refers to its endpoint, so the
comparison time is **printed time + tau**. All site-zero records are retained;
the latest duplicate at each physical time is compared with ED, and duplicate
spread is recorded. Endpoint clocks are separately validated in HDF5. A last impurity measurement must not be identified with the final saved
endpoint, which can occur at the other lattice boundary.

For T=0.6, tau=0.2/0.1/0.05 need respectively 4/6/12 directional legs to cover
the comparison grid. The process may compute later target times; comparisons
are restricted to the declared horizon. This is an explicit accounting of the
observed schedule, not an assumption that tau times the loop count equals the
last measured impurity time.

## What ED can say about tau

For the zero-bath-energy N7 control, ED gives initial energy -1, energy standard
deviation 0.72266615, and reference-sector energies approximately
[-2.02937421,1.02937421], in units with hbar=1. Thus tau=0.2 corresponds to
sigma_E*tau=0.145 and a full-sector bandwidth phase span about 0.612 radians.
This makes 0.2 a reasonable scale to test, not a demonstrated optimal timestep.
The 0.1 and 0.05 controls test sensitivity at the same physical horizon.

Exact constant-H propagation itself has no time-discretization error from
sampling at those grids. Comparing the DMRG results at matched times tests the
projection/advancement algorithm and numerical propagation. These scales do
not bound the sampling error of an arbitrary rapidly changing hybridization.

Constant connectors and degenerate bath energies admit collective-mode
simplifications. A nonzero bath bandwidth control removes that particular
simplification; it still is not a decomposed self-consistent DMFT bath. Neither
static control, even if exact, alone resolves the changing-connector discrepancy
in the earlier experiments.

## Measured results

N7, U=2, m=256, fixed bath energies uniformly spaced from -1 to +1,
advanceEach=5. These are the largest absolute complex errors across the alpha
lesser and beta greater **t'=0 amplitudes at the common final time T=0.6**:

| tau | two target times | three target times (engine default) |
|---:|---:|---:|
| 0.2 | 1.21e-5 | 3.58e-9 |
| 0.1 | 3.08e-6 | 1.92e-10 |
| 0.05 | 9.70e-7 | 1.61e-11 |
| 0.025 | 2.67e-7 | not run |

Increasing m to 512 with two target times gave 1.23e-5, 3.09e-6, and 9.71e-7
at tau=0.2,0.1,0.05: little benefit compared with refining tau or using the
third target time. Zero bath energies gave comparable two-target errors.

With three targets, maximum paired-reference impurity observable errors across
the sampled grid were 3.56e-9, 1.04e-10, and 7.77e-12 respectively. All these
normal-schedule runs reported zero discarded weight and no WFT warnings.
Thus discarded weight alone misses significant time-target/projection error.
Do not infer a formal convergence order from these combined schedule and
projection tests.

For these static controls, tau=0.1 with three targets is already very accurate;
0.05 provides a useful refinement check. This is **not** a universal tau choice
for a time-dependent self-consistent bath. The earlier changing-connector,
frozen-window protocol has not been revalidated here. Its 1e-4 plateau should
not be interpreted as an intrinsic engine accuracy floor.

The three-target runs used about 102-103 MiB peak child RSS and 3.5-6.3 seconds
summed over six stage processes per tau (ED excluded). Retained spaces are
intentionally much larger than the original low-m stress cases.

## Reproduction and inputs

From the source root, with the same activated build environment and Python
dependencies as README:

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python \\
  cincuenta/TestSuite/tdmrg_reference/run_normal_schedule.py \\
  --executable build/dmrg/dmrg --output build/normal_new_run \\
  --nsites 7 --m 256 --bath-bandwidth 2 \\
  --taus .2,.1,.05 --final-time .6 --target-times 3
```

Use a fresh output root. `--target-times 2` reproduces the minimal-target
controls; `--m 512` gives the retained-space check. No production engine code
was changed.

Local generated evidence, not tracked in Git:

* `build/tdmrg_normal_w2_targets3/`: default-target control and full raw logs.
* `build/tdmrg_normal_w2_m256/`: two-target nondegenerate control.
* `build/tdmrg_normal_w0_m256/`: two-target degenerate-bath control.
* `build/tdmrg_normal_w2_m512/`: two-target retained-space check.
* `build/tdmrg_normal_w2_tau025/`: finer two-target step.

For example, inspect these actual three-target inputs:

* `build/tdmrg_normal_w2_targets3/tau_0.1/alpha_seed/alpha_seed.ain`
* `build/tdmrg_normal_w2_targets3/tau_0.1/alpha_insert/alpha_insert.ain`
* `build/tdmrg_normal_w2_targets3/tau_0.1/alpha_continuous/alpha_continuous.ain`

This is a static paired-reference test, not an independently evolved
reference-only trajectory, full two-time matrix test, or bath-decomposition
bootstrap. Only the initially nonzero t'=0 channel of each seed is probed.

Validation: all 22 Python unit tests and 12 focused engine CTests passed;
logs are `build/tdmrg-normal-unit.log` and `build/tdmrg-normal-ctest.log`.
Independent review found no confirmed correctness defect in the advancing
runner or the three-target endpoint/clock generalization.
