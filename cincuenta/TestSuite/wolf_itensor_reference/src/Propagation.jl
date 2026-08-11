"""One-step real-time propagation primitives.

Time-grid interpolation and factorization are intentionally outside this file.
A caller must supply the Hamiltonian evaluated at the desired midpoint, making
that numerical choice explicit and independently testable.
"""

using ITensorMPS: MPS, tdvp

"""Evolve `state` by one real-time TDVP step using an explicit midpoint MPO.

`midpoint_hamiltonian` must represent the Hamiltonian at the caller-chosen
midpoint. The returned state approximates
`exp(-im * midpoint_hamiltonian * dt) * state`. This function neither reads a
model configuration nor chooses/interpolates a factorization-grid time.

A zero step returns a copy without invoking TDVP. `nsite`, truncation, and
bond-dimension controls remain ordinary keyword arguments rather than hidden
solver state.
"""
function midpoint_tdvp_step(
        state::MPS,
        midpoint_hamiltonian,
        dt::Real;
        cutoff::Real = 1e-12,
        maxdim::Integer = typemax(Int),
        nsite::Integer = 2,
        normalize::Bool = false,
        kwargs...,
    )
    isfinite(dt) || throw(ArgumentError("dt must be finite"))
    cutoff >= 0 || throw(ArgumentError("cutoff must be nonnegative"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    nsite in (1, 2) || throw(ArgumentError("nsite must be 1 or 2"))
    iszero(dt) && return copy(state)

    return tdvp(
        midpoint_hamiltonian,
        -im * dt,
        state;
        nsteps = 1,
        cutoff,
        maxdim,
        nsite,
        normalize,
        kwargs...,
    )
end

"""Take one explicit midpoint TDVP step and measure its returned state.

`midpoint_hamiltonian` and `impurity_position` are supplied by the caller, so
this helper has no hidden grid, factorization, or model lookup. It merely
composes the independently testable TDVP primitive with `state_diagnostics`.
The returned `NamedTuple` contains the new state, the caller's `dt`, and
measurements made using that same midpoint Hamiltonian.
"""
function midpoint_tdvp_observed_step(
        state::MPS,
        midpoint_hamiltonian,
        dt::Real,
        impurity_position::Integer;
        kwargs...,
    )
    evolved = midpoint_tdvp_step(state, midpoint_hamiltonian, dt; kwargs...)
    return (
        state = evolved,
        dt = dt,
        diagnostics = state_diagnostics(evolved, midpoint_hamiltonian, impurity_position),
    )
end
