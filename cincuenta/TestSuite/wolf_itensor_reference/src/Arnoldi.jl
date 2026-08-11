"""Global Arnoldi-basis construction for explicitly compressed MPS operations.

This module constructs Krylov vectors from repeated global MPO actions. It
intentionally does not exponentiate the projected matrix or advance time;
those are separate steps so the basis construction can be tested on its own.
Every MPS action and subtraction receives explicit compression controls.
"""

using LinearAlgebra
using ITensorMPS: MPS

function _validate_arnoldi_controls(
        krylovdim::Integer,
        action_cutoff::Real,
        action_maxdim::Integer,
        orthogonalization_cutoff::Real,
        orthogonalization_maxdim::Integer,
        breakdown_tolerance::Real,
    )
    krylovdim > 0 || throw(ArgumentError("krylovdim must be positive"))
    _validate_compression(action_cutoff, action_maxdim)
    _validate_compression(orthogonalization_cutoff, orthogonalization_maxdim)
    isfinite(breakdown_tolerance) && breakdown_tolerance >= 0 ||
        throw(ArgumentError("breakdown_tolerance must be finite and nonnegative"))
    return nothing
end

"""Construct an explicitly compressed global MPS Arnoldi basis.

The input state is copied and normalized to form `q₁`. At each column, an MPO
is applied globally, then full modified Gram--Schmidt plus one explicit
reorthogonalization pass is performed. `basis` contains exactly the accepted
Krylov vectors used for a projected exponential; `next_vector` is the
normalized residual direction for the final column when it exists.

`hessenberg` has shape `(accepted_dimension + 1, accepted_dimension)`. The
last row is the residual coefficient and remains zero on breakdown. The two
`*_defects` matrices record overlaps with the current basis after the first
and second subtraction passes, respectively. They expose loss of
orthogonality from compressed MPS arithmetic instead of silently discarding
it.
"""
function global_mps_arnoldi_basis(
        state::MPS,
        hamiltonian;
        krylovdim::Integer,
        action_cutoff::Real,
        action_maxdim::Integer,
        orthogonalization_cutoff::Real,
        orthogonalization_maxdim::Integer,
        breakdown_tolerance::Real,
    )
    _validate_arnoldi_controls(
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        breakdown_tolerance,
    )

    basis = MPS[normalized_mps(state)]
    hessenberg = zeros(ComplexF64, krylovdim + 1, krylovdim)
    first_pass_coefficients = zeros(ComplexF64, krylovdim, krylovdim)
    second_pass_coefficients = zeros(ComplexF64, krylovdim, krylovdim)
    first_pass_defects = zeros(ComplexF64, krylovdim, krylovdim)
    second_pass_defects = zeros(ComplexF64, krylovdim, krylovdim)
    next_vector = nothing
    breakdown = false
    accepted_dimension = 0

    for column in 1:krylovdim
        current_basis = basis[1:column]
        vector = mps_action(
            hamiltonian,
            basis[column];
            cutoff = action_cutoff,
            maxdim = action_maxdim,
        )

        first_coefficients = mps_basis_overlaps(current_basis, vector)
        first_pass_coefficients[1:column, column] = first_coefficients
        vector = mps_subtract_projection(
            vector,
            current_basis,
            first_coefficients;
            cutoff = orthogonalization_cutoff,
            maxdim = orthogonalization_maxdim,
        )
        first_pass_defects[1:column, column] = mps_basis_overlaps(current_basis, vector)

        second_coefficients = mps_basis_overlaps(current_basis, vector)
        second_pass_coefficients[1:column, column] = second_coefficients
        vector = mps_subtract_projection(
            vector,
            current_basis,
            second_coefficients;
            cutoff = orthogonalization_cutoff,
            maxdim = orthogonalization_maxdim,
        )
        second_pass_defects[1:column, column] = mps_basis_overlaps(current_basis, vector)
        hessenberg[1:column, column] = first_coefficients + second_coefficients

        residual_norm = mps_norm(vector)
        accepted_dimension = column
        if residual_norm <= breakdown_tolerance
            breakdown = true
            break
        end

        hessenberg[column + 1, column] = residual_norm
        next_vector = vector / residual_norm
        column == krylovdim || push!(basis, next_vector)
    end

    dimension = accepted_dimension
    return (
        basis = basis,
        next_vector,
        hessenberg = hessenberg[1:(dimension + 1), 1:dimension],
        first_pass_coefficients = first_pass_coefficients[1:dimension, 1:dimension],
        second_pass_coefficients = second_pass_coefficients[1:dimension, 1:dimension],
        first_pass_defects = first_pass_defects[1:dimension, 1:dimension],
        second_pass_defects = second_pass_defects[1:dimension, 1:dimension],
        accepted_dimension = dimension,
        breakdown,
        input_norm = mps_norm(state),
    )
end
