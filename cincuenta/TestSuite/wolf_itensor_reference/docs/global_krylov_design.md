# Global MPS Krylov midpoint propagation: design

Status: the basis builder and a single midpoint-step primitive are implemented
and tested on a static `Lb=10` model. The remote first-order noninteracting
amplitude acceptance test is also complete. This document specifies the
independently testable implementation needed to reproduce the propagation class
used by Wolf, McCulloch, and Schollwoeck. The finite-time covariance and
numerical-convergence acceptance tests remain outstanding.

## Why this replaces local TDVP

Wolf et al. state that they use an **MPS Krylov time-evolution algorithm** and
a midpoint step

\[
 |\psi(t+\Delta t)\rangle =
 \exp[-iH(t+\Delta t/2)\Delta t]|\psi(t)\rangle.
\]

See `neqdmft.tex`, around lines 922--933.  Their discussion of the bond
dimensions of the *set of Krylov states* (around lines 1029--1034) is
consistent with global MPS Krylov vectors, not local TDVP Krylov updates.

The shelved two-site TDVP prototype was unable to create a first-order remote
star-leg transfer from the rank-one product state.  A global Krylov vector
`Hmid * psi` contains every such transfer before MPS compression, which is the
property required here.

## Explicit inputs and outputs

The implemented one-step function has this interface:

```julia
midpoint_global_krylov_step(
    state, midpoint_hamiltonian, dt;
    krylovdim,
    action_cutoff, action_maxdim,
    orthogonalization_cutoff, orthogonalization_maxdim,
    combination_cutoff, combination_maxdim,
    breakdown_tolerance,
)
```

It returns a data-only tuple:

```text
(state, diagnostics)
```

The caller supplies the midpoint MPO and every numerical control.  The step
must not read a configuration, interpolate a factor row, select a time grid,
or retain a hidden trajectory state.

## Algorithm

For a Hermitian midpoint MPO `H` and normalized input `q₁`, construct an
Arnoldi basis (rather than assuming exact Lanczos tridiagonality):

1. `w = apply(H, q_j; cutoff=action_cutoff, maxdim=action_maxdim)`.
2. For every already accepted `q_i`, compute `h[i,j] = <q_i|w>` and subtract
   `h[i,j] q_i`, using an explicit MPS sum/compression policy.
3. Reorthogonalize once, recording the second-pass overlaps.
4. Set `h[j+1,j] = ||w||`; stop on a specified breakdown threshold, otherwise
   normalize `w` to make `q_{j+1}`.
5. Exponentiate the resulting small dense Hessenberg matrix,
   \(y=\exp(-i\Delta t h)e_1\), using ordinary dense linear algebra.
6. Form `sum_j y[j] q_j` with the separately explicit combination-compression
   policy.  Do not normalize merely to hide norm loss.

The MPS action and sums are truncated approximations, so they are not exactly
linear.  For that reason this implementation must own its Arnoldi loop rather
than pass a truncated `apply` closure directly to `KrylovKit.exponentiate`,
whose mathematical contract assumes a linear map.

The current ITensorMPS `expand(state, H; alg="global_krylov")` is useful
basis-expansion machinery: it applies `H` repeatedly and adds orthogonal
basis support while preserving `state`.  It is **not** a complete exponential
propagator with the numerical controls and residual report required here.

## Required diagnostics

Each step must expose, rather than conceal:

- accepted Krylov dimension and breakdown status;
- projected Hessenberg matrix and its anti-Hermitian defect
  `norm(h - h')`;
- first- and second-pass orthogonality defects;
- projected exponential residual estimate
  \( |h_{m+1,m} e_m^T\exp(-i\Delta t h_m)e_1| \);
- input/output norm and norm drift;
- all action, orthogonalization, and combination truncation settings.

The projected residual alone is not a total error bound because each MPS
contraction/sum can be truncated.  Convergence must therefore vary Krylov
dimension, all three compression controls, and `dt` independently.

## Acceptance sequence

1. **Algebraic unit tests:** explicit MPS addition/scaling/overlap, Arnoldi
   orthogonality, and a static-`Lb=10` midpoint-step norm/diagnostic test are
   complete.
2. **Remote-amplitude test:** complete at `Lb=10`, `U=0`. A down electron is
   transferred from the remote left bath site to an initially-Up impurity. The
   selected target-state amplitude agrees at order `dt` with its direct MPO
   matrix element, and its scaled first-order error decreases under timestep
   refinement. This is more sensitive than an occupation, whose leading
   change is order `dt^2`.
3. **Finite-time free-fermion test:** compare impurity occupations and double
   occupancy to `FreeFermion.jl` for the same centered-star finite bath,
   factor grid, and midpoint Hamiltonians.
4. **Numerical convergence:** require stable agreement under simultaneous
   `dt`, Krylov-dimension, and MPS-compression refinement before moving to
   `U=4` or `U=10`.

No `Lb<10` MPS input is permitted in these tests.  Passing the free-fermion
gate validates the propagator plumbing only; it is not a Wolf benchmark or a
bath-convergence claim.
