# neq-DMFT C++ development roadmap

## Scope and current checkpoint

The [design contracts](neq_tdmrg_design.md) are the agreed baseline. Production
work belongs in MPI C++ under `cincuenta/src`; Python is a validation oracle,
not a second production DMFT implementation. Traditional renormalized-block
DMRG is the target engine, not a proposed MPS/MPO redesign.

**Next action:** pause t-DMRG implementation and carry out the
[ED acceptance refactor](neq_ed_acceptance_handoff.md) in a separate branch and
session. That refactor must be validated before resuming t-DMRG development.
This roadmap does not authorize starting the backend during that session.
For a later fresh session after the ED exit gate, use the
[resume handoff](neq_tdmrg_resume_handoff.md) and the completed ED refactor's
actual return report; the original planning conversation is not required.

The design baseline settles transaction ownership, interval locality, adapter
responsibilities, and cincuenta MPI ownership. Concrete signatures,
initialization transitions, class decomposition, and MPI scheduling details
remain implementation-stage decisions; they are not evidence of an already
implemented backend. Open t-DMRG-specific details do not block the bounded ED
lifecycle refactor.

## E — Establish the lifecycle in the existing ED backend

Separate session, serial/root-only execution:

- [ ] Record numerical and test baselines before editing.
- [ ] Define the explicit lifecycle and n=0 initialization; distinguish accepted
      progress from candidate propagation/cache validity.
- [ ] Add meaningful `beginTimeStep`/`acceptTimeStep` operations to the ED backend
      and make `NeqDmftSolver` invoke them explicitly.
- [ ] Track candidate validity across bath preparation, both active spin seeds,
      states, Green-function results, and diagnostics. Audit decomposition
      bookkeeping, not just the current V row.
- [ ] Test retries, bath replacement, acceptance guards, and driver call order.
- [ ] Establish numerical baseline equivalence and independently review the
      result. Explain and separately review any necessary numerical-policy
      change rather than hiding it in the structural refactor.
- [ ] Commit and return a handoff describing the resulting shared contract.

**Exit gate:** reviewed C++ ED lifecycle with reproducible validation. No ED MPI
support, changed backward-row algorithm, DMRG engine work, or interpolation
change. Stop that session here. Resuming t-DMRG is a subsequent work decision.

## T1 — C++ DMRG execution boundary

After the ED exit gate and an explicit decision to resume:

- [ ] Implement typed run descriptions and Ainur generation for initialization,
      insertion, and paired evolution; keep logical state roles separate from
      engine P-vector slots.
- [ ] Invoke the existing in-process `DmrgRunner`, with explicit application
      context and isolated run artifacts; no production Python subprocess driver.
- [ ] Parse and validate structured measurements, checkpoint metadata, clocks,
      and available norm/truncation diagnostics.
- [ ] Keep the known insertion workaround behind one replaceable policy point.
      Coordinate an engine fix with Gonzalo or implement it as separate work;
      use the [bug handoff](../TestSuite/tdmrg_reference/PVECTOR_INSERTION_BUG.md).
- [ ] Validate static C++ execution against ED and existing reference fixtures.
      Use the normal-advancement fixtures (`run_normal_schedule.py`) for this
      control. The older frozen-window harness deliberately suppresses advance
      and is not a production default to copy. Missing measurements, mismatched
      clocks, and failed runs must fail loudly.

**Exit gate:** actual C++ engine execution with checked complex results. Do not
inherit from the equilibrium `ImpuritySolverBase` or transplant correction-vector
physics. Three responsibilities do not mandate three classes. No general new
DMRG two-time input language is required.

## T2 — Changed-Hamiltonian propagation and trial replay

- [ ] Define checkpoint handles and accepted/trial artifact ownership against
      the established lifecycle.
- [ ] Implement the initial frozen-midpoint interval coefficient policy,
      matching the C++ ED Hamiltonian and complex-hopping conventions.
- [ ] Establish how the engine refreshes/rebuilds affected cached block
      Hamiltonians. New input connector values alone are not proof of correctness.
- [ ] Validate prescribed complex time-dependent bath propagation, nonzero-time
      creation/annihilation insertions, and rejected/corrected trial replay.
- [ ] Check that accepted representations remain unmodified. Never reuse a
      representation transformed to follow a superseded trial endpoint.

**Exit gate:** changing-bath and replay agreement with ED under the same
interval convention. Static-bath agreement does not satisfy this gate.

## T3 — Serial neq t-DMRG backend and feedback

- [ ] Implement the neq solver-template contract; reuse the existing real-time
      target, bath decomposition, GF container, and progressive corrector loop.
- [ ] First validate the atomic, half-filled pure-state path with the required
      incoherent alpha/beta averaging; do not silently substitute product states
      for a correlated first-bath initialization.
- [ ] Produce the complete causal row using forward insertion trajectories,
      both channels, and explicit equal-time handling. Preserve branch weights.
- [ ] Choose and document the reference defining equal-time observables; monitor
      reference spread when separate paired jobs evolve approximate references.
- [ ] Validate several accepted time steps and multiple correctors against ED,
      including norms, complex GFs, equal-time identities, and diagnostics.
- [ ] Add explicit backend selection and supported-mode validation without
      changing the legacy ED scientific algorithm.
- [ ] Audit inherited parameter coupling: the neq backend must not acquire an
      equilibrium Matsubara/correction-vector dependency through convenience
      helpers or the current parameter layout.

**Exit gate:** initial self-consistent C++ t-DMRG path in serial with demonstrated
transaction behavior. No general thermal/correlated-first-bath support is implied.

## T4 — Cincuenta-owned MPI execution

ED remains serial/root-only. MPI work applies to t-DMRG only:

- [ ] Verify rank-local `DmrgRunner` execution/communicator behavior for real-time
      targeting before running independent jobs concurrently.
- [ ] Choose a simple initial ownership/control protocol. Decide which small
      data are replicated, how candidate revisions are communicated, and whether
      stable deterministic job assignment is adequate before considering a
      dynamic scheduler or dedicated coordinator.
- [ ] Distribute candidate-row trajectory work with distinct task identities and
      artifacts; allow sequential small-target runs per worker to limit memory.
- [ ] Assemble complete rows with coverage and candidate-identity checks.
      Coordinate acceptance and failure handling without collective deadlocks.
- [ ] Compare one and multiple ranks against the serial backend, including
      uneven/empty worker assignments and failed/incomplete trial handling.

Insertion-time parallelism here refers to the proposed forward paired-state
algorithm, with retained accepted trajectories and potentially redundant
reference propagation. It is not a parallelization of ED's sequential backward
Krylov sweep. No next candidate/physical step may outrun its causal dependencies.

**Exit gate:** rank-count-consistent feedback and safe collective transaction
handling. Generalizing DMRG input or MPI orchestration remains deferred.

## T5 — Numerical validation and later extensions

- [ ] Vary time resolution, target sampling, retained states, and bath rank
      separately. Do not equate small discarded weight with propagation accuracy.
- [ ] Include larger baths/retained spaces as feasible; tiny unit systems are
      correctness tests, not evidence of physical DMFT convergence.
- [ ] Measure per-rank process memory, checkpoint volume, runtime, and replay
      overhead; identify temporary checkpoint cleanup policy after validation.
- [ ] Compare to Wolf's reported data with documented protocol differences,
      not a claim of reproducing its underspecified interpolation/MPS algorithm.

Optional later projects: interval-local spline/Hermite coefficients with a
matching propagator and ED oracle; correlated first-bath state preparation;
more efficient scheduling/checkpoint transport; general two-time engine-level
coordination/input. Each needs its own scope decision. None is a prerequisite
for the ED acceptance refactor.
