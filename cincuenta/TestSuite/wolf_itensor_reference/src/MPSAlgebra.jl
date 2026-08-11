"""Explicit MPS algebra used by the custom global-Krylov implementation.

These small wrappers make every compression decision an argument. They are
separate from the eventual Arnoldi loop so MPO action, MPS summation, and
projection can be tested independently on the `Lb=10` technical model.
"""

using LinearAlgebra
using ITensors: inner
using ITensorMPS: MPS, apply

function _validate_compression(cutoff::Real, maxdim::Integer)
    isfinite(cutoff) && cutoff >= 0 ||
        throw(ArgumentError("cutoff must be finite and nonnegative"))
    maxdim > 0 || throw(ArgumentError("maxdim must be positive"))
    return nothing
end

"""Return `H * state` with caller-selected MPO-application compression."""
function mps_action(hamiltonian, state::MPS; cutoff::Real, maxdim::Integer)
    _validate_compression(cutoff, maxdim)
    return apply(hamiltonian, state; cutoff, maxdim)
end

"""Return the ordinary Hilbert-space overlap of two MPS states."""
mps_overlap(left::MPS, right::MPS) = inner(left, right)

"""Return the MPS norm without modifying the supplied state."""
mps_norm(state::MPS) = norm(state)

"""Return a normalized copy of `state`, rejecting the zero vector."""
function normalized_mps(state::MPS)
    state_norm = mps_norm(state)
    iszero(state_norm) && throw(ArgumentError("cannot normalize a zero MPS"))
    return copy(state) / state_norm
end

"""Form `sum(coefficients[j] * states[j])` under explicit MPS-sum compression."""
function mps_linear_combination(
        states::AbstractVector{<:MPS},
        coefficients::AbstractVector{<:Number};
        cutoff::Real,
        maxdim::Integer,
    )
    length(states) == length(coefficients) ||
        throw(ArgumentError("states and coefficients must have equal length"))
    isempty(states) && throw(ArgumentError("at least one state is required"))
    all(isfinite, coefficients) ||
        throw(ArgumentError("coefficients must be finite"))
    _validate_compression(cutoff, maxdim)

    weighted_states = [coefficient * copy(state) for (state, coefficient) in zip(states, coefficients)]
    return +(weighted_states...; cutoff, maxdim)
end

"""Subtract explicit basis projections from `vector` with explicit compression.

The result represents `vector - sum(coefficients[j] * basis[j])`. Callers
compute and retain the coefficients, so Arnoldi/reorthogonalization diagnostics
are not hidden inside this helper.
"""
function mps_subtract_projection(
        vector::MPS,
        basis::AbstractVector{<:MPS},
        coefficients::AbstractVector{<:Number};
        cutoff::Real,
        maxdim::Integer,
    )
    length(basis) == length(coefficients) ||
        throw(ArgumentError("basis and coefficients must have equal length"))
    states = MPS[vector]
    append!(states, basis)
    return mps_linear_combination(
        states,
        vcat(ComplexF64[1], -ComplexF64.(coefficients));
        cutoff,
        maxdim,
    )
end

"""Return overlaps of `vector` with an explicit ordered basis."""
mps_basis_overlaps(basis::AbstractVector{<:MPS}, vector::MPS) =
    ComplexF64[mps_overlap(state, vector) for state in basis]
