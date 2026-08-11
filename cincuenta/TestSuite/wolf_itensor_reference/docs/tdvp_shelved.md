# Shelved local-TDVP experiment

This validation is a short, `Lb=10` **technical check**, not a Wolf benchmark
result and not evidence of bath-size convergence.  Its exact reference is a
one-particle calculation for the same finite factorized bath at `U=0`.

## Current result: TDVP is not yet accepted for the centered star

The first attempted comparison deliberately exposed a problem rather than
producing a passing validation.  In the centered star ordering, an initially
occupied bath site on the left of the impurity is non-nearest-neighbor.  The
MPO has a nonzero matrix element for its down-spin hop into the impurity, but
a two-site TDVP sweep from the product state did not create that remote
particle-hole fluctuation over the tested two steps.  In contrast, the
nearest-neighbor up-spin process from the impurity into the first empty bath
site on its right did evolve.

This is consistent with a two-site TDVP tangent-space/projector limitation
for this long-range star ordering, not evidence that the MPO term is absent: a
direct MPO matrix element test is nonzero. Raising `maxdim`, setting the
cutoff to zero, and increasing the local Krylov dimension did not change the
observation. The centered-star TDVP propagator is therefore **not validated
for production use**. A controlled ordering/basis-expansion test is still
needed before calling this mechanism proven or discarding every TDVP variant.

## Why this is independent

For `U=0`, the spin sectors are independent and the factorized SIAM is a
quadratic Hamiltonian.  For either spin,

\[
H(t) = \sum_{ij} h_{ij}(t)c_i^\dagger c_j,
\]

where the single-particle index order is exactly the MPS order: bath sites,
the central impurity, then the remaining bath sites.  `FreeFermion.jl` builds
`h` directly from the data-only model specification.  It does **not** reuse
the MPO or MPS propagator.

It uses the one-particle density-matrix convention

\[
\rho_{ij}=\langle c_j^\dagger c_i\rangle.
\]

For a midpoint interval of duration `dt`, its exact update is

\[
\rho(t+dt)=e^{-i h(t+dt/2)dt}\,\rho(t)\,
            e^{+i h(t+dt/2)dt}.
\]

The reference now selects a global MPS Krylov propagator, following Wolf et
al. This record is retained because the same exact covariance calculation will
be its first noninteracting acceptance test. Any later TDVP experiment must use
the same caller-selected midpoint coupling row, finite-bath factorization, and
atomic `|up>` state; it may not be treated as an approved propagator merely
because it has an MPO API.

## What is compared

For each endpoint of a short two-step run, the unit test compares:

- impurity `Nup` and `Ndn` from the MPS against the diagonal of the exact
  up- and down-spin density matrices;
- impurity double occupancy against
  \(n_\uparrow n_\downarrow\), valid because the two `U=0` spin sectors
  remain independent for the product initial state;
- total particle number, spin projection, and the bounds
  \(0\le d\le1\).

The test also verifies that the exact covariance propagation preserves each
spin-sector particle number.  It intentionally does not compare MPS and
covariance energy: a time-dependent Hamiltonian has different endpoint and
midpoint energy conventions, while occupations directly exercise the target
operator and state conventions.

## Limits

This check validates the implementation only for the supplied finite bath and
short interval. It cannot validate interacting (`U=4` or `U=10`) dynamics,
the provisional ramp duration, bath-factorization convergence, late-time
recurrences, or agreement with Wolf et al.'s figure.
