# Operator-on-P-vector overwrite: engine handoff and workaround removal

**Status:** reproduced in the current engine. Not fixed here. The initial
cincuenta implementation must not silently rely on operator-on-P-ket insertion
until this regression and nonzero-time restart checks pass.

This is an expression-evaluation defect independent of restart: after applying
an operator to a P ket, a stale source P index can cause the result to be
replaced with the original source vector. It can therefore retain the wrong
particle sector and give zero instead of a nonzero Green-function amplitude.

## Focused reproducer

```sh
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
DMRG_EXECUTABLE="$PWD/build/dmrg/dmrg" \
  build/experiment-venv/bin/python -m unittest discover \
  -s cincuenta/TestSuite/tdmrg_reference \
  -p test_pvector_insertion_bug.py -v
```

The test creates two independent N5/m64 jobs, with U=0, zero potentials and
couplings, and initial product state `[2,3,3,0,0]`. It materializes a nonempty
reference target with:

```text
string P0="TimeEvolve{tau=.1,steps=3,advanceEach=3,EnergyForExp=0}*|gs>";
```

H=0 means this target is the unchanged product state. A bare `P0=|gs>` does
not materialize the required nonempty P target and is not a valid substitute
in this reproducer. The two jobs differ only in their inserted state:

```text
# Control
string P1="c[0]*|gs>";
# Regression
string P1="c[0]*|P0>";
```

Verified results:

| Measurement | Correct/control | Current direct-P result |
|---|---:|---:|
| `<P1|c|P0>` | 1 | 0 |
| `<P1|P0>` | 0 | 1 |
| `<P1|P1>` | 1 | 1 |

The last row illustrates why a norm check alone misses this defect. Persistent
local evidence is in `build/tdmrg_bug_adapter_repro/{control_time,bug_time}/`.
The opt-in unittest runs the same cases in fresh project-local temporary
folders. Without DMRG_EXECUTABLE it skips the engine tests. Job setup and
measurement extraction occur outside the expected-failure assertion, so a
missing measurement or failed process is an error, not a false reproduction.
A separate ordinary test requires the direct result to match either the known
bug signature or the correct signature; other numerical failures cannot hide
behind the expected-failure marker.

## Source trace for Gonzalo

* `dmrg/Engine/KetForTargetingExpression.hh`, `multiply`: applying an operator
  changes the ket kind to M but retains its old source `pIndex_`.
* `dmrg/Engine/TermForTargetingExpression.h`, `finalize` / `pIndex`: after the
  operators have been consumed, the term can expose that stale source index.
* `dmrg/Engine/TargetingExpression.h`, `computePvectors`: the final single-P-term
  assignment executes `tvNonConst(i)=tv(x)`, replacing the operator result.
  That assignment also copies the source time label.

A fix must distinguish a genuine bare-P assignment from an operator result;
ordinary P-vector copies, factors, and TimeEvolve expressions must still work.
No particular implementation of that fix is prescribed here.

## Isolated caller workaround

`insertion_restart.py::plan_insertion` is the **only insertion-policy point**.
For a saved reference it currently selects `gs-promotion-workaround`: load the
saved endpoint as gs through RestartSourceTvForPsi, leave the new branch
unmapped, and apply c/c' to gs. This resets the local clock to zero. The returned
InsertionPlan owns the ket roles, mappings, expected insertion clock, and
physical clock origin. `Case.insert_branch` executes the plan; the Green-function
loop consumes those fields instead of knowing how promotion works.

Initial-time insertion on the actual gs is named `initial-gs` and is not a
workaround. `run_normal_schedule.py` uses that natural initial-state path and
needs no workaround removal. Summaries explicitly record which insertion
backend was used. No silent engine-version detection or fallback is used.

## Fix acceptance and removal checklist

1. Fix the engine defect and run the focused reproducer. Its known-bug test uses
   `unittest.expectedFailure`: a fixed engine produces **unexpected success**,
   deliberately failing the suite until that decorator is removed. Retain it
   thereafter as an ordinary passing regression.
2. Verify creation as well as annihilation, genuine P copies/factors, and
   TimeEvolve. In particular, verify that an operator on a saved P endpoint
   preserves the **physical insertion time**. The zero-H/time-zero reproducer
   above establishes state correctness, not that clock requirement.
3. Replace only the saved-reference branch of `plan_insertion` with a validated
   direct-P implementation. One candidate layout keeps branch P0 and reference
   P1: mapping `[-1, saved_endpoint]`, `P0=c[0]*|P1>` (or c'), `P1=|P1>`;
   no RestartSourceTvForPsi. Return first-pair mapping `(0,1)`,
   `first_pair_from_gs=False`, physical insertion time as the local insertion
   time, and clock origin zero. Validate the ordering and actual engine result;
   this layout currently has **mocked contract coverage only**, not a live
   supported mode.
4. Compare the new backend against the old promotion backend and ED on the
   full N5 two-time suite, including nonzero insertion times, both channels,
   norms, sectors, clocks, and replay. Then remove promotion-specific code and
   tests, keeping initial-gs insertion and the backend-neutral driver contract.

`test_insertion_restart.py` checks the present adapter and a mocked physical-clock
backend contract. The refactored N5 suite retains exactly the previous complex
Green matrices and reference observations; evidence is in
`build/tdmrg_insertion_adapter/n5_m64/summary.json` compared with
`build/tdmrg_validated/n5_m64/summary.json`.

Validation: 30 opt-in tests ran, with 29 passing and the one reproduced engine
bug marked expected failure (`build/tdmrg-insertion-unit.log`). Exact numerical
before/after comparisons are recorded in `build/tdmrg-insertion-equivalence.json`.
Independent review found no blocking adapter or reproducer issue.
