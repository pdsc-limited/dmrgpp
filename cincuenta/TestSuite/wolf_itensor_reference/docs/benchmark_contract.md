# Benchmark contract: Wolf et al. non-self-consistent SIAM

This contract defines the first target for this independent reference.  It is
not a contract for NEQ-DMFT self consistency and is intentionally more
specific about claims than the paper's prose permits.

## Target

Reproduce the non-self-consistent SIAM test in Sec. “Results for a
non-self-consistent impurity problem” of Wolf, McCulloch, and Schollwoeck,
PRB **90**, 235131 (2014).  The target hybridization is

\[
\Lambda(t,t') = v(t)g(t,t'),\qquad
 g^{\gtrless}(t,t') = \mp i\int d\omega\,
 f^{\gtrless}(\omega) A(\omega)e^{-i\omega(t-t')},
\]

with

\[
A(\omega)=\frac{\sqrt{4-\omega^2}}{2\pi}\quad (|\omega|\le2),
\qquad \beta=1,
\]

and the cosine ramp

\[
v(t)=\begin{cases}
[1-\cos(\pi t/t_1)]/2,&t<t_1,\\
1,&t\ge t_1.
\end{cases}
\]

The observable is impurity double occupancy for `U=4` and `U=10`, in units
where `v0 = hbar = kB = 1`.  The paper reports agreement between `Lb=20` and
`Lb=22` to approximately `t=11`.

Primary source anchors: paper source `neqdmft.tex`, Eqs. `eqSiam`,
`eqHybDecomp`, `eqRamp`, and `eqLambdaSimple`--`eq:bathgf` (respectively near
lines 230, 324, 886, and 1363 in the supplied arXiv source).

## Finite bath represented by a time-grid factorization

For this target, `Lb` means the number of **spinful bath sites**, is even,
and is split evenly between the initially occupied and initially empty
sectors.  It is not a number of spin orbitals.

On a chosen real-time grid, form the Hermitian positive-semidefinite kernels

\[
K^< = -i\Lambda^<,\qquad K^> = i\Lambda^>.
\]

Their rank-`Lb/2` approximations define coupling columns

\[
K^< \approx V_{\rm occ}V_{\rm occ}^\dagger,\qquad
K^> \approx V_{\rm empty}V_{\rm empty}^\dagger.
\]

Equivalently, `Lambda^< = i Vocc Vocc†` and
`Lambda^> = -i Vempty Vempty†`.  The bath sites have homogeneous final
potential zero, with the two sectors initialized occupied and empty.  This
is the paper-style finite-bath dilation of the actual `beta=1` lesser and
greater kernels.  A factorized occupied/empty product state is *not* a
zero-temperature replacement: its finite-temperature content is in the
factorized kernels and must be checked by reconstructing them.

A direct spectral quadrature is permitted only as a later, independent
reference.  Equal-size agreement with such an ED bath is not evidence of
convergence to this benchmark.

### Midpoint convention

For midpoint time propagation, the factorization grid contains both the
propagator endpoints and their midpoints.  The two kernels are factorized once
on that joint grid, and a midpoint coupling is read from its corresponding
factor row directly.  The implementation must not independently factorize
endpoint and midpoint grids, nor silently interpolate rows from independent
factorizations: their column gauges need not agree. The planned uniform grid
has equal quadrature weights; a nonuniform grid requires an explicit weighted
factorization and validation of that changed fitting norm. The selected
propagator is a global MPS Krylov approximation to the explicit midpoint MPO,
as used by Wolf et al.; local TDVP and TEBD are deferred.

## Allowed sizes and claim gates

No MPS calculation, application input, or regression input may use `Lb<8`.
The executable benchmark contract is stricter: all runner configurations
must use even `Lb>=10`.  The initial sequence is `10, 12, 14, 16`; the paper
comparison target is `20, 22`.  The runner must reject smaller sizes.

No result may be called a reproduction until all of the following hold:

1. the `K^<` and `K^>` reconstruction errors and discarded spectral weights
   are recorded for every `Lb` and time grid;
2. the source of the numerical ramp duration `t1` is recorded;
3. the impurity initial-state protocol is recorded with its source evidence;
4. timestep and bond-dimension/cutoff convergence are documented; and
5. the `Lb=20`/`22` trajectory comparison is made through the claimed time.

## Deliberately unresolved historical inputs

The supplied paper source specifies `t1>0` but not its numerical value.
Therefore neither `dt` nor `tmax` is chosen in the baseline configurations.

The numerical `t1` is the remaining source-level ambiguity.  The cited predecessor,
Balzer *et al.*, arXiv:1407.6578, Sec. II.C.1 (archived locally at
`/workspace/dmrgpp_papers/mctdh impurity solver balzer et al 2015/`), resolves
the state convention: the atomic impurity is singly occupied by an up- or
down-spin electron, while the first half of spinful bath sites are doubly
occupied and the second half empty.  The spin-symmetric implementation will
run the two atomic impurity components and average scalar observables.  Thus
`d(0)=0`.  This is the finite-temperature-kernel dilation described above,
not a Gibbs state of the finite SIAM.  A full atomic Gibbs state is deliberately
out of scope for the paper-protocol trajectory and must be named separately if
implemented later.

Balzer's representative non-self-consistent SIAM uses a Heaviside switch-on;
it therefore does **not** provide the numerical `t1` for Wolf et al.'s cosine
ramp.  For implementation and numerical development, use the explicitly
provisional engineering default `t1=0.25` from the independent GBEK cosine
ramp inputs (for example,
`cincuenta/TestSuite/inputs/inputNeqAtomicLimitGBEKL3.ain`).  This keeps the
ramp smooth rather than treating the protocol as a Heaviside quench.  It is
not source evidence for Wolf's value and cannot support a paper-reproduction
claim until independently resolved.

## Scope boundary

This benchmark takes `Lambda(t,t') = v(t)g(t,t')v(t')` as an input.  It does
not update a DMFT hybridization, compute a lattice Green function, or perform
self consistency.
