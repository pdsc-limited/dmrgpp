# Resume handoff: C++ neq t-DMRG after the ED acceptance refactor

## When to use this

Use this document in a **new session after** the separate ED lifecycle refactor
has been validated and reviewed. It replaces the need to load the earlier
planning conversation. It is not permission to skip the ED prerequisite or to
resume automatically at the end of that refactoring session.

At writing, the planning checkpoint is `bcf36cbd`; no C++ t-DMRG backend or
explicit ED acceptance lifecycle has been implemented by the planning session.
The ED refactor's future branch/commits/API/results are deliberately not guessed
here. Read its actual return handoff and source before proceeding.

## Start with these checks

1. Read the worktree's AGENTS/overlay instructions and verify revision, branch,
   dirty state, and active Spack environment. Preserve unrelated work.
2. Obtain the ED refactor's return handoff: commit range, lifecycle API/state
   table, n=0 handling, bath-bookkeeping decisions, test/baseline results,
   numerical differences, and unresolved issues.
3. Verify that the chosen validated ED commits are present in the working
   source. If they are not, ask which commits/worktree to use or integrate;
   do not redo the ED refactor or silently start from the old implicit API.
4. Read [neq_tdmrg_roadmap.md](neq_tdmrg_roadmap.md). Confirm the ED exit gate
   and select a bounded next task, normally T1. Consult
   [neq_tdmrg_design.md](neq_tdmrg_design.md) for the detailed agreed contracts.
5. If the ED refactor is incomplete, unreviewed, or has unresolved numerical
   changes, stop and resolve that prerequisite with the user.

The ED assignment itself is [neq_ed_acceptance_handoff.md](neq_ed_acceptance_handoff.md).
It is historical scope guidance, not a substitute for the completed refactor's
report. No original conversation history is required.

## Settled decisions — do not reopen by default

- Production is MPI C++ in `cincuenta/src`, not Python orchestration in
  `TestSuite/tdmrg_reference`. Python remains an independent back-checking tool.
- Reuse the **real-time** neq driver, hybridization target, bath decomposition,
  and GF container. Do not transplant equilibrium Matsubara/correction-vector
  algorithms or inherit from the equilibrium `ImpuritySolverBase` for helpers.
- Traditional renormalized-block DMRG is the engine. No MPS/MPO engine redesign.
- The DMFT coordinator explicitly begins and accepts steps. Every candidate
  repropagates from unchanged accepted history at n-1, including the compatible
  renormalized representations—not ones transformed while following a rejected
  trial. A better restart mechanism would not remove this transaction.
- Match ED's **frozen midpoint** coefficients initially. Accepted interval
  coefficient functions are immutable. Future interpolation options are assumed
  interval-local; no spline or new interpolation keyword is selected now.
- Keep coefficient policy, numerical propagation, and target-time sampling
  separate. Three targets do not mean three integration stages.
- The DMRG execution boundary has three responsibilities: generate a typed run,
  execute through in-process `DmrgRunner`, and read/validate structured results.
  Class count remains open; do not invent a universal base hierarchy.
- Cincuenta owns MPI. Ranks execute separate ordinary DMRG runs; no new general
  single-input multi-insertion engine facility or nested MPI launches now.
- Candidate-row parallelism refers to forward paired insertion trajectories
  under one fixed bath revision, not ED's backward Krylov sweep. Reference
  propagation can be redundant; independent truncated bases cannot be compared
  by an ordinary coefficient-vector overlap. Monitor reference spread and
  explicitly choose the reference defining equal-time observables.
- ED stays serial/root-only. No ED MPI optimization or row-algorithm replacement.
- Wolf comparison means validation against reported data, not exact reproduction
  of underspecified interpolation or its different MPS-formalism implementation.

## Relevant C++ boundaries

Inspect current source, especially changes from the ED refactor:

- `cincuenta/src/NeqDmftSolver.h`: shared progressive SCF/lifecycle owner.
- `cincuenta/src/ImpuritySolverNeqGBEK.h`: ED oracle and scientific conventions.
- `cincuenta/src/NeqBathDecomposition.h`, `NeqHybridizationTarget.h`,
  `NeqRealTimeGf.h`: reusable real-time infrastructure.
- `cincuenta/src/cincuenta.cpp`: backend dispatch; ED currently root-only.
- `dmrg/Engine/DmrgRunner.h` and implementation: existing C++ engine entry.
- `cincuenta/src/ImpuritySolverDmrg.h`: invocation mechanics reference only;
  its equilibrium input generation and frequency processing are not the neq API.
- `ParamsNeqDmftSolver.h`: audit the existing physical/Matsubara parameter
  coupling rather than giving the new backend a dummy frequency-grid dependency.

The first validation scope proposed in the roadmap is the atomic, half-filled
pure-state path with alpha/beta averaging. Do not silently claim support for
correlated first-bath or general finite-temperature initialization.

## Evidence available, and its limits

Committed reference utilities are in `cincuenta/TestSuite/tdmrg_reference`.
Read their README selectively; do not turn them into the production backend.

- Normal advancement uses `advanceEach=Nsites-2`; the engine default is three
  target times. A static nondegenerate N7/m256 control at tau=0.1 gave about
  1.9e-10 complex GF error at T=0.6; tau=0.05 gave about 1.6e-11.
- Those controls validate static propagation, not changing-bath restart or
  self-consistent feedback. The old two-target, advancement-suppressed harness
  is a stress fixture, not a production default or evidence of an engine floor.
- An operator-on-P-ket overwrite defect remains known at this checkpoint. Read
  [the bug handoff](../TestSuite/tdmrg_reference/PVECTOR_INSERTION_BUG.md).
  A gs-promotion workaround is isolated in the prototype insertion adapter;
  it resets local time. A future C++ adapter must isolate it similarly, or use
  a separately validated engine fix. Do not assume the ED refactor fixed it.
- Changed input connectors alone are not proof that checkpointed internal block
  Hamiltonians have been refreshed. Establish that path before claiming
  changing-bath correctness.

Generated pilot artifacts may remain in the original checkout under
`build/tdmrg_normal_w2_targets3` and related directories, but are not tracked or
portable prerequisites. Do not copy build caches/checkpoints with old absolute
paths. Regenerate focused evidence in the new worktree's build area if needed.
The planning session did not run a C++ ED baseline; use the completed refactor's
actual validation report, not old DMRG/Python test counts.

## What remains deliberately open

The roadmap supplies task gates, not a complete implementation specification.
Resolve choices within the selected task: class decomposition, typed request/
result details, artifact retention/transport, safe Hamiltonian refresh, and later
MPI control/data ownership and scheduling. Verify rank-local engine execution;
unrelated jobs must not accidentally participate in MPI world collectives.

Prefer simple small-target calculations and redundant work over high memory per
rank. Validate sequential C++ execution before MPI and feedback complexity.
Changing-bath replay and nonzero-time insertions require their own ED comparisons.
Do not treat static agreement or tiny discarded weight as sufficient evidence.

Follow the project-local build/temp and Spack policy, commit coherent validated
work, stage only task-owned changes, obtain scoped independent review, and
consult before pushing. No branch/worktree creation or push was performed by
writing this resume handoff.

## Opening prompt for the new session

> Resume C++ neq t-DMRG planning/development using
> `cincuenta/doc/neq_tdmrg_resume_handoff.md`. First read the completed ED
> acceptance refactor's return handoff and verify its validated commits are in
> this worktree. Then propose the first bounded task from the roadmap against
> the actual implemented lifecycle. Keep Python as an oracle, ED serial, and
> equilibrium correction-vector physics out of the neq backend. Do not restart
> the ED refactor or reopen settled decisions without a concrete new issue.
