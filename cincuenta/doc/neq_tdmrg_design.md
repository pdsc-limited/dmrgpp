# Non-equilibrium t-DMRG: agreed design contracts

**Status:** design for a C++ impurity backend in `cincuenta/src`; not a statement
that this backend or a spline option is implemented. The current production
neq solver is `ImpuritySolverNeqGBEK` (ED). Python reference calculations remain
back-checking tools, not the production neq-DMFT infrastructure.

## Reuse the real-time DMFT layer

The intended implementation reuses `NeqDmftSolver`, `NeqHybridizationTarget`,
`NeqBathDecomposition`, and `NeqRealTimeGf`. It replaces the ED-specific impurity
propagation with traditional renormalized-block DMRG. It does not port the
scientific algorithm from the equilibrium Matsubara/correction-vector solver.
Existing in-process `DmrgRunner` invocation and MPI programming patterns can be
mechanical references without importing frequency-domain physics.

## Accepted history and corrector trials

At physical step n, the SCF correctors update the complete new target row

    Lambda(t_n,t_j) = v(t_n) Gimp(t_n,t_j) v(t_j),  j = 0,...,n,

and fit the corresponding bath endpoint amplitudes V(t_n). Here v is the
lattice hopping; V denotes impurity-to-auxiliary-bath hopping amplitudes.
Previously accepted time rows and endpoint amplitudes remain fixed.

Every candidate for step n repropagates from the same accepted state at
`t_(n-1)`. For DMRG, this means the accepted reference/inserted states **and their
renormalized representations**, not versions transformed or truncated while
following a preceding trial at `t_n`. Repropagation is an intentional cost of
this algorithm; trial evolution is not assumed reversible.

**Interval-locality contract:** once an interval is accepted, its bath
coefficient function is immutable. A corrector may change the current trial
interval but must not retroactively change the Hamiltonian on accepted
intervals. Acceptance freezes both the endpoint data and the chosen interval
representation. Any future scheme requiring changes to accepted intervals
would require a different rollback algorithm and is outside this contract.

### Explicit lifecycle shared by ED and t-DMRG

The transaction belongs to self-consistent real-time DMFT, not merely to a
particular DMRG checkpoint format. Even a more elegant t-DMRG restart mechanism
would not remove the distinction between accepted history and candidate
propagation. Both impurity backends should expose explicit `beginTimeStep(n)`
and `acceptTimeStep(n)` operations, controlled by `NeqDmftSolver`.

The intended control flow is:

```text
beginTimeStep(n)             # accepted prefix still ends at n-1
compute predictor G row
repeat configured correctors:
    construct Lambda row from current trial G
    prepareTimeStep(n, Lambda)  # replace candidate bath; invalidate trial propagation
    computeGimp(n)             # evaluate from accepted n-1, never implicitly accept
acceptTimeStep(n)            # publish the completed, matching bath/state/G result
```

Acceptance must not perform an additional bath update. It accepts the candidate
bath and the propagation/observables evaluated with that candidate, not a new
fit for which no propagation has been performed. Fixed-count correctors do not
thereby become an exactly converged fixed-point solve. An incomplete or failed
trial must not become the starting point of the next physical step.

The accepted/trial boundary includes state validity, bath-decomposition
bookkeeping, Green-function rows, and diagnostics—not only state-vector files.
Detailed lifecycle signatures, initialization at n=0, and failure/cancellation
handling remain part of the remaining interface design.

#### Existing implicit ED behavior

`ImpuritySolverNeqGBEK::prepareTimeStep` already invalidates propagation through
n by reducing `propagatedThrough` to at most n-1. The next `computeGimp` calls
`ensurePropagated`, repropagates the stored reference from n-1 with the revised
midpoint Hamiltonian, regenerates inserted states at n, and overwrites the
candidate Green-function row and diagnostics. Accepted earlier history is
retained. The last candidate at n is implicitly accepted when propagation to
n+1 uses it as its starting state.

This is easier in ED because the Fock-space basis is fixed: the stored vector
at n-1 stays usable after a trial. DMRG must also retain its compatible
renormalized representation. The current ED implementation nevertheless has
no explicit accepted-step marker or general failure-safe transaction; candidate
bath data, state entries, and outputs are updated in place.

The planned ED lifecycle is therefore meaningful, not just no-op compatibility
methods for a DMRG interface. It can use the existing history arrays and
lightweight acceptance/validity bookkeeping rather than copying the entire ED
history for every trial. The treatment of mutable decomposition bookkeeping
must be audited as part of that refactor.

## Bath coefficients within an interval

The initial implementation must match the current C++ ED convention:

    V_mid = (V(t_(n-1)) + V(t_n)) / 2.

`ImpuritySolverNeqGBEK::computeVMid` computes these amplitudes and holds them
fixed over the interval. The DMRG backend must match the same Hamiltonian,
complex-hopping orientation, and occupied/empty-bath conjugation conventions.
Changing V(t_n) during a corrector therefore requires recomputing that interval.
Changing input connectors alone must not be assumed to refresh all Hamiltonian
terms cached in a DMRG checkpoint; the implementation must establish and test
the update/rebuild path.

The convention must be isolated behind an interval coefficient policy, so a
future alternative need not alter Green-function assembly or SCF control flow.
The initial policy is **frozen midpoint**. Alternate policies are assumed to be
**interval-local**, not global refits of the accepted coefficient history.
A future cubic/Hermite or spline policy must specify its endpoint derivatives,
boundary conditions, and available data, and preserve accepted intervals.
Merely calling a scheme "piecewise" does not guarantee locality: coefficients
of an ordinary piecewise cubic spline can depend on samples across the entire
trajectory.

No particular spline construction or new input keyword is selected here.
Midpoint behavior and its ED comparisons are the initial implementation target.

### Separate coefficient policy, propagation, and targeting

Keep these distinct in the C++ design:

1. The bath coefficient function supplied over the physical interval.
2. The numerical propagator that integrates that time-dependent Hamiltonian.
3. The intermediate state samples used in DMRG targeting.

Three TimeEvolve targets mean states at the window start, midpoint, and end;
they are not automatically three integration stages or three evaluations of V.
For frozen midpoint all three states follow the same constant Hamiltonian.
For a future varying coefficient function, target states must follow the
appropriate time-ordered propagation. Independently substituting a different
instantaneous Hamiltonian into each constant-H exponential is not generally
equivalent.

The physical DMFT grid spacing, engine propagation window, number of target
times, sweep advancement, and local/physical checkpoint clocks must also remain
explicit rather than being treated as interchangeable controls.

## C++ DMRG execution boundary

Three responsibilities are agreed; their division into classes is not yet
chosen:

1. **Describe and generate an engine run.** Translate a typed description of
   the interval Hamiltonian, initialization/insertion/evolution operation,
   checkpoint sources, logical state roles, propagation settings, and requested
   measurements into DMRG input. Keep Ainur syntax, PureState encoding, complex
   serialization, P-vector indices, and the temporary insertion workaround
   inside this boundary.
2. **Execute the run.** Invoke the existing in-process `DmrgRunner` with an
   explicitly supplied application context and unique run/artifact identity.
   This responsibility does not perform SCF, acceptance, bath decomposition,
   or MPI scheduling.
3. **Read and validate the result.** Return structured complex measurements,
   checkpoint/state-slot metadata, physical/local clocks, and available norm
   and truncation diagnostics. Existing logs and checkpoint metadata are an
   acceptable initial transport; missing requested measurements or inconsistent
   clocks must fail explicitly. Reading engine output is not Green-function
   assembly or an acceptance decision.

The neq backend chooses the required calculation; this boundary expresses and
executes it; `NeqDmftSolver` owns the transaction/SCF control flow. Python inputs
provide tested examples and regression fixtures, not a production dependency.
MPI ownership belongs to cincuenta, as specified below; supplying a unique
job identity to this boundary does not imply that the boundary schedules ranks.

### Equilibrium interfaces are not neq base classes

`ImpuritySolverBase` is the base of the **equilibrium** impurity solvers. Its
frequency-domain interface, Matsubara/real-frequency state, spectral-weight
processing, and GS/omega input-generation assumptions do not define a neq
impurity solver. The new backend must not inherit from it merely to obtain
input-string helpers. `ImpuritySolverDmrg` is likewise not the scientific or
architectural parent of the neq backend.

The existing `NeqDmftSolver` solver-template contract, extended by the agreed
explicit lifecycle, is the intended integration point. Reuse `DmrgRunner`
itself and applicable low-level invocation patterns. Leave the equilibrium
classes unchanged initially; any later extraction of genuinely shared helpers
should be justified by concrete duplication rather than by their current names.

### Class decomposition remains open

Do not equate three responsibilities with a mandatory three-class hierarchy.
They may initially be cohesive methods/private helpers with explicit request
and result records; class extraction should follow actual ownership, testing,
and lifetime requirements. The direct LanczosPlusPlus integration in
`ImpuritySolverNeqGBEK` and the runner-mediated DMRG interface have different
boundaries. The shape of that existing ED header alone does not settle the
appropriate number of classes for the new backend. No new universal impurity
base class is selected by this design.

## MPI ownership and scope

**Cincuenta owns MPI coordination for the initial neq t-DMRG implementation.**
Its ranks execute separate ordinary DMRG calculations through the C++ execution
boundary. The initial parallelism is across trajectory/insertion jobs, not a
requirement to distribute one target vector across those workers. Cincuenta
owns their dependencies, trial revisions, result assembly, and acceptance.

The general need for multiple insertions is not specific to DMFT: a full
G(t,t') for a prescribed finite chain also needs such calculations. A general
single-input DMRG facility and reusable MPI coordination may therefore be useful
later. They are deliberately **not prerequisites** for this work. Do not expand
DMRG's input language or introduce nested MPI launches to solve that broader
problem now. Use the existing engine input vocabulary for each individual run.
After the cincuenta case is understood, consider whether its coordination can
be factored into a general facility and what user-facing input that requires.

### Coordination requirements

* Use explicit logical identities for trajectory jobs and their accepted
  source/trial bath revision. Physical task identity must not be only an MPI
  rank number. Assignment policy and checkpoint retention are separate choices.
* Give runs distinct artifact namespaces so independent ranks/trials cannot
  overwrite one another. Accepted representations remain immutable under the
  shared transaction contract.
* All jobs contributing to a candidate row must use the same identified bath
  candidate and accepted prefix. Assemble the required channels and spin-seed
  weights with explicit coverage checks; missing results are not zero values.
* Acceptance is coordinated by cincuenta. A rank cannot independently promote
  its trial while another required job has failed or is incomplete. Errors
  must be communicated without leaving other ranks blocked in collectives.
* Preserve the low-memory intent: a worker can run its assigned small target
  sets sequentially. This design does not require collecting all insertion
  targets into one DMRG calculation or keeping all their vectors in memory.

### Details still to settle

Cincuenta ownership alone does not select static versus dynamic job assignment,
replicated versus coordinator-owned small control data, result collectives,
artifact sharing/movement, or a dedicated coordinator rank. These remain design
choices, not claims of implemented MPI support.

Also verify that a worker's `DmrgRunner` invocation is rank-local. Unrelated
jobs must not accidentally enter engine collectives over the same MPI world
communicator. Existing independent-run usage is a useful reference, not proof
that every real-time targeting path has the required communicator behavior.
If isolation is needed, establish the appropriate runtime context without
turning this into a new general DMRG input design.

## Meaning of a comparison with Wolf

For this project, the proposed Wolf comparison is **validation against reported
data**, not a claim of exact algorithmic reproduction. The available description
of "standard spline interpolation" does not establish the needed boundary,
causal-support, or coefficient-update details. We must not infer from that
phrase whether a global or progressive bath decomposition was used, or assert
that our future interval-local interpolation is identical to theirs.

Wolf's MPS-formalism implementation is also distinct from the traditional
renormalized-block DMRG engine used here. We can compare matched physical
models, observables, and controlled numerical results without claiming the same
state-evolution or truncation algorithm. Document known protocol differences
and uncertainty when making such comparisons.

The first implementation-level oracle remains our C++ ED solver with matching
midpoint coefficients. Interpolation changes should be checked against a
matching ED propagation convention before paper-data discrepancies are
attributed to DMRG.

## Roadmap and session boundary

The agreed order is:

1. **Finish this design.** The C++ DMRG execution boundary's responsibilities
   and separation from equilibrium interfaces are agreed above; concrete class
   decomposition remains open. MPI ownership is assigned to cincuenta; the
   concrete execution/coordination policy remains to be settled. Complete shared
   lifecycle details as needed, including initialization, candidate validity,
   and acceptance.
2. **Pause neq-DMFT t-DMRG development after the design is complete.** Do not
   proceed directly from this discussion into a new t-DMRG backend.
3. **In a separate session, refactor the existing C++ ED implementation to the
   explicit acceptance lifecycle.** Update `NeqDmftSolver` and
   `ImpuritySolverNeqGBEK`, including bath-decomposition bookkeeping as needed.
   Keep this separate from DMRG engine or backend implementation work.
4. Validate the ED refactor before resuming t-DMRG implementation against that
   established lifecycle. Resumption is a subsequent work decision, not an
   automatic continuation of the ED refactor.

The ED refactor should preserve the current midpoint convention, supported
physical initialization, spin averaging, Green-function conventions, and
fixed-corrector scheme. Establish numerical baseline equivalence; any discovered
transaction defect requiring numerical changes must be explained and reviewed,
not hidden inside a structural refactor. Tests should cover repeated candidate
evaluation from unchanged accepted history, replacement of a candidate bath,
explicit acceptance, rejection of invalid step transitions, and prevention of
accepting incomplete or mismatched trial results.

## Remaining implementation planning

These contracts do not yet settle the detailed C++ backend API, accepted/trial
artifact management, concrete MPI task assignment and coordination,
initial-state support beyond the atomic pure states, or the changing-Hamiltonian
checkpoint mechanism. The known
operator-on-P-vector insertion defect also remains unfixed; its temporary
adapter must stay isolated. See
[`../TestSuite/tdmrg_reference/PVECTOR_INSERTION_BUG.md`](../TestSuite/tdmrg_reference/PVECTOR_INSERTION_BUG.md)
for the reproducer and fix/removal checklist.
