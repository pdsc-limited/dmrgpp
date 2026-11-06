# Session handoff: explicit acceptance lifecycle for neq-DMFT ED

## Assignment and stopping point

In a **separate branch/worktree and session**, refactor the existing C++
non-equilibrium ED implementation to make its accepted/trial lifecycle explicit.
The shared scientific design is in [neq_tdmrg_design.md](neq_tdmrg_design.md);
sequencing and later work are in [neq_tdmrg_roadmap.md](neq_tdmrg_roadmap.md).

**Do this refactor only. Stop after validation, review, commits, and a return
handoff. Do not continue into the t-DMRG backend.** The user explicitly requested
a pause in neq t-DMRG development for this separate ED session.

Start from a commit containing this handoff and the design decisions. The source
inventory was made at `1a7b8a92`; preceding planning/reference commits have not
implemented the lifecycle in C++. Verify the actual starting revision and dirty
state; do not assume later changes or unrelated user edits can be overwritten.
No new worktree or branch was created by the session writing this handoff.

## Non-goals

- No MPI parallelization of ED. Keep serial/root-only neq ED execution.
- No DMRG backend, engine changes, P-vector bug fix, or new DMRG input format.
- No Python production DMFT infrastructure.
- No equilibrium Matsubara/correction-vector refactor or new universal impurity
  base class. `ImpuritySolverBase` is equilibrium-specific.
- No replacement of ED's backward Krylov row algorithm with forward insertion
  jobs, new interpolation scheme, adaptive SCF algorithm, or physics extension.

## Current C++ behavior

`NeqDmftSolver::solve` initializes the backend, computes G at zero, constructs
its hybridization target, and calls `prepareTimeStep(0, ...)`.
For n>0 it first calls `computeGimp(n)` **before any bath preparation at n**,
using the existing zero-row cold start for V[n]. Preserve this predictor;
`beginTimeStep` must establish a valid initial candidate without requiring a
fit first. The driver then performs exactly `neqDmftIter` correctors:

```text
update complete Lambda(n,j), j=0,...,n
prepareTimeStep(n, Lambda)
computeGimp(n)
```

The final G is evaluated with the final candidate bath. Acceptance is implicit:
the next step uses the last computed state at n. Do not append another bath fit
at acceptance or silently claim that a fixed number of correctors converged.
The stored target used for the last fit need not equal a target reconstructed
from the final G at finite corrector count.

`ImpuritySolverNeqGBEK::prepareTimeStep` updates the decomposition, then lowers
each active spin sector's `propagatedThrough` to at most n-1. `computeGimp`
lazily repropagates from the stored n-1 reference, reseeds inserted states at n,
computes diagnostics, and fills the GF row. The fixed Fock basis makes retaining
that old reference easy, but does not eliminate the transaction semantics.

## Required lifecycle

The driver owns explicit `beginTimeStep(n)` and `acceptTimeStep(n)` calls.
Between them, candidate bath preparation and G evaluation must not implicitly
accept anything. Every revised candidate starts from unchanged accepted history.

```text
beginTimeStep(n)
evaluate predictor
repeat existing correctors:
    construct Lambda from current trial G
    prepare candidate bath; invalidate candidate propagation/results
    compute matching trial G and diagnostics from accepted n-1
acceptTimeStep(n)
```

Acceptance publishes a complete, mutually consistent candidate bath/state/G/
diagnostic result, not merely whatever happens to occupy a cache. Starting the
next physical step must not implicitly commit the previous candidate. A failed,
incomplete, or stale evaluation must not become accepted.

Use meaningful acceptance/validity bookkeeping, not no-op hooks. Retain existing
history arrays where possible; copying the entire Fock-space history per trial
is not required. Candidate revision IDs or an equivalent explicit validity
mechanism should make stale-result checks testable.

### Details to settle at the start of the refactor

Document a small state-transition table before coding:

- How n=0 initialization becomes accepted. `decomp.update(0, ...)` is currently
  a no-op; ordinary n>0 propagation assumptions must not be applied blindly.
- Preconditions for begin, prepare, compute, and accept; accepted progress must
  be distinct from propagation/cache progress.
- Behavior for repeated compute of an unchanged candidate, prepare after a
  computed candidate, premature acceptance, skipped/backward steps, and begin
  while another step is active. Prefer explicit guards over compatibility
  fallbacks that silently begin/accept steps.
- Whether a cancellation operation is needed now, or only invalidation/retry
  with accepted state preserved. Do not invent a general restart framework.
- Ownership of provisional G/diagnostics and what dumps may expose. Trial data
  needed for the next corrector may exist without being accepted history.

## Mutation inventory and pitfalls

Inspect these together:

| State | Current location / relevance |
|---|---|
| Reference histories and insertions | `ExtendedSector::PhiNHist`, `bStates`, `dStates` for alpha and, when distinct, beta |
| Propagation validity | Each sector's `propagatedThrough`; currently means cached propagation, not explicit acceptance |
| Hamiltonian workspaces | N/N-1/N+1 CSR matrices and mutable hopping entries; backward sweeps also update workspaces |
| Bath state | `NeqBathDecomposition::V_` **and `maxDiagSeen_`**, not just V[n] |
| Results | Driver G row/conjugate column, `doccHistory_`, `ekinHistory_`, and target row used for the fit |
| Published extent | Dumps currently use propagated progress; audit accepted versus provisional output |

`maxDiagSeen_` controls the Cholesky seeding floor. An intervening candidate can
therefore affect more than the visible current V row. Audit candidate versus
accepted bookkeeping explicitly. Do not silently alter regularization to force
a desired rollback result: if the stronger lifecycle exposes history-dependent
numerical behavior, report it and obtain agreement on the numerical-policy
change, separately from structural refactoring.

Do not change the midpoint convention `(V[n-1]+V[n])/2`, centered impurity
Hamiltonian, complex hopping/conjugation conventions, spin-seed averaging,
branch amplitudes, GF signs, supported initialization, rank bootstrap, or fixed
corrector order/count as incidental cleanup. Accepted interval coefficients
remain immutable. No spline implementation is required.

## Source and test map

Read first:

- `cincuenta/src/NeqDmftSolver.h`: shared driver/template contract and call order.
- `cincuenta/src/ImpuritySolverNeqGBEK.h`: backend lifecycle/cache, propagation,
  backward GF rows, diagnostics, and dump extent.
- `cincuenta/src/NeqBathDecomposition.h`: candidate row and persistent bookkeeping.
- `cincuenta/src/NeqHybridizationTarget.h`, `NeqRealTimeGf.h`: target and output
  conventions; reuse rather than redesign.
- `cincuenta/src/cincuenta.cpp`: current root-only neq dispatch; retain it.
- `cincuenta/src/tests/test_ImpuritySolverNeqGBEK.cpp`,
  `test_NeqBathDecomposition.cpp`, `test_NeqRealTimeGf.cpp`, `test_NeqGbekInput.cpp`.
- Test registration: `cincuenta/src/tests/CMakeLists.txt` and
  `cincuenta/src/CMakeLists.txt`.
- Input/physics guide: [neq_dmft_ed_input.md](neq_dmft_ed_input.md).

Existing tests include private-state/direct-Hamiltonian fixtures and independent
Python-derived numerical tables. Audit direct calls to prepare/compute and any
direct decomposition mutation: adapt fixtures to the explicit lifecycle without
weakening their scientific assertions or leaving production auto-accept paths.
There is no dedicated driver lifecycle mock test in the inspected tree. Use the
existing `NeqDmftSolver<..., ImpSolverTemplate>` seam to add an instrumented
backend test of call order and acceptance; a new inheritance framework is not
needed.

Useful existing inputs:

- `cincuenta/TestSuite/inputs/inputNeqGBEKFast.ain`: fast non-atomic first-bath
  plus positive-rank ED end-to-end regression.
- `cincuenta/TestSuite/inputs/inputNeqAtomicLimitGBEKL3.ain`: atomic-limit coverage.

The ED refactor must preserve both existing supported paths, even though the
first proposed t-DMRG validation path is atomic.

## Build environment and baseline procedure

Follow the worktree's AGENTS/overlay instructions. Start through the paired
DMRG++ overlay; do not add AGENTS.md or .pi to the source repository. If creating
a worktree, choose the inherited context explicitly; do not copy a CMake build
cache or checkpoints containing old absolute paths.

Use the activated project Spack environment; the preparation session used
`/workspace/spack_env/scientific-workstation`. Build and keep temporary data
inside the new checkout, normally `build` and `build/tmp`. Do not use global
temporary directories. If dependencies do not satisfy the build, stop and
resolve that with the user rather than silently switching environments.

Example serial configuration/targets (confirm against the new checkout):

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DDMRG_BATCHED_GEMM_TYPE=CPU -DDMRGPP_USE_MPI=OFF \
  -DCMAKE_CXX_COMPILER=clang++
cmake --build build --target cincuenta test_ImpuritySolverNeqGBEK \
  test_NeqBathDecomposition test_NeqRealTimeGf test_NeqGbekInput -j3
```

Run the Catch executables directly for reliable selection:

```sh
build/cincuenta/src/tests/test_ImpuritySolverNeqGBEK
build/cincuenta/src/tests/test_NeqBathDecomposition
build/cincuenta/src/tests/test_NeqRealTimeGf
build/cincuenta/src/tests/test_NeqGbekInput
ctest --test-dir build -R '^neqGBEKFast(_check)?$' --output-on-failure
```

Individual Catch tests are registered under case names, not necessarily their
executable names: a CTest regex matching only `test_ImpuritySolverNeqGBEK` can
select nothing. Verify selection with `ctest -N`. The fast end-to-end tests are
conditional on Python discovery; their checker uses stdlib only. New lifecycle
and driver tests must also be built and run.

**Validation status at handoff:** no C++ ED build/test baseline was executed in
this documentation session. Prior DMRG/Python experiment tests do not establish
an ED-refactor baseline. Run and record the above before editing, using isolated
output locations for numerical before/after comparisons. Avoid changing an
existing equilibrium regression merely to make a neq lifecycle test pass.

## Acceptance and return handoff

Require tests for:

- Correct driver begin/corrector/accept ordering, including initialization.
- Repeated unchanged-candidate evaluation and replacement after evaluation.
- No mutation of accepted prefix; recomputation from accepted n-1.
- Matching accepted bath revision, both active spin sectors, G, and diagnostics.
- Rejection of premature/stale acceptance and invalid time-step transitions.
- Retry/failure safety to the defined scope; no published partial accepted state.
- Atomic and non-atomic numerical baseline comparisons, including full complex
  GFs and bath/observable outputs rather than occupation checks alone.

Keep numerical algorithms unchanged where possible. Explain any baseline drift
and separate numerical fixes from interface/state-management work. Prepare the
changed-file inventory and focused test results before independent review.
Commit coherent validated blocks locally, stage only task-owned changes, and
discuss before pushing.

Return the commit range, chosen API/state table, baseline and post-change test
commands/results, numerical differences, unresolved issues, and integration
instructions. Update the shared design/roadmap to match the implemented contract.
**Stop; do not resume t-DMRG implementation in this session.**

## Suggested opening prompt for the new session

> Read `cincuenta/doc/neq_ed_acceptance_handoff.md` and the linked design.
> Implement only the explicit accepted/trial lifecycle for the existing C++
> neq-DMFT ED backend and driver. Establish baselines first; preserve serial
> execution and the current scientific algorithm. Do not implement t-DMRG or
> ED MPI. Validate, obtain independent review, commit locally, and stop with a
> return handoff.
