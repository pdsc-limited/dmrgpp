# ITensorMPS research notes

Research date: 2026-08-10.  This records public-source findings before
implementing the Wolf--McCulloch--Schollwoeck non-self-consistent SIAM
reference.

## Current Julia package split

- `ITensors.jl` v0.9 provides core tensors, indices, and site types.
- `ITensorMPS.jl` v0.4 provides MPS/MPO algorithms; MPS/MPO functionality
  moved there from `ITensors.jl` and the former `ITensorTDVP.jl`.
- The project must explicitly depend on both packages and import them with
  `using ITensors, ITensorMPS`.

Sources:

- <https://docs.itensor.org/ITensorMPS/stable/>
- <https://docs.itensor.org/ITensors/stable/>

## Directly reusable API foundations

`siteinds("Electron", N; conserve_qns=true)` provides the spinful fermion
site type.  Its documented/source operators include `Nup`, `Ndn`, `Nupdn`,
`Ntot`, `Cup`, `Cdagup`, `Cdn`, and `Cdagdn`.  The SIAM Hamiltonian can use
an `OpSum` and `MPO(os, sites)`.  An MPO may contain arbitrary impurity--bath
two-site terms, so a star geometry needs no new MPO representation.

Official MPS time-evolution material demonstrates TEBD via a gate list and
`apply(gates, psi; cutoff)`, including non-nearest-neighbor gates.  This is
useful background only: the current reference selects a Wolf-style global MPS
Krylov propagator. Local TDVP is shelved after failing its centered-star
noninteracting validation, and TEBD is deferred.

Sources:

- <https://docs.itensor.org/ITensorMPS/stable/tutorials/MPSTimeEvolution.html>
- <https://github.com/ITensor/ITensors.jl/blob/main/src/lib/SiteTypes/src/sitetypes/electron.jl>
- <https://github.com/ITensor/ITensorMPS.jl/blob/main/examples/solvers/01_tdvp.jl>

## Finite temperature: useful but not solved by an example

ITensorMPS includes a `finite_temperature/purification.jl` example.  It
starts from an infinite-temperature density MPO, applies imaginary-time
Trotter gates, and normalizes with `tr(rho)`.  It is a spin-chain density-MPO
example, not a fermionic thermofield/purified SIAM implementation.  It does
not decide the Wolf benchmark's bath discretization, electron ancilla
ordering, Fermi occupations, or its initial thermal state.  Those remain
explicit design/validation tasks.

Source:

- <https://github.com/ITensor/ITensorMPS.jl/blob/main/examples/finite_temperature/purification.jl>

## Third-party code assessment

- `mtfishman/ITensorImpurity.jl` is an old experimental, zero-temperature
  Anderson-transport example.  It targets ITensors 0.2--0.3, uses a small
  chain (`N=8` in its example), and is not a Wolf/NEQ-DMFT implementation.
  Do not use it as a dependency or physics template.
- `angusdunnett/MPSDynamics` is a more recent thermofield/TDVP codebase, but
  uses TensorOperations.jl rather than ITensorMPS.  It may inform algorithmic
  choices only; it is not a drop-in component.
- No public Julia/ITensor code reproducing Wolf et al., PRB 90, 235131, was
  found in the research pass.  This is not proof that none exists.

Sources:

- <https://github.com/mtfishman/ITensorImpurity.jl>
- <https://github.com/angusdunnett/MPSDynamics>
- <https://link.aps.org/doi/10.1103/PhysRevB.90.235131>
