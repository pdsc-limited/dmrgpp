"""One midpoint global-MPS Krylov step built from explicit Arnoldi vectors.

The MPO action and every MPS compression remain caller-controlled. This module
owns the small dense projected exponential but deliberately does not manage a
time grid, bath row, or trajectory.
"""

using LinearAlgebra
using ITensorMPS: MPS

function _validate_global_krylov_controls(
        dt::Real,
        krylovdim::Integer,
        action_cutoff::Real,
        action_maxdim::Integer,
        orthogonalization_cutoff::Real,
        orthogonalization_maxdim::Integer,
        combination_cutoff::Real,
        combination_maxdim::Integer,
        breakdown_tolerance::Real,
    )
    isfinite(dt) || throw(ArgumentError("dt must be finite"))
    _validate_arnoldi_controls(
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        breakdown_tolerance,
    )
    _validate_compression(combination_cutoff, combination_maxdim)
    return nothing
end

function _global_krylov_controls(
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        combination_cutoff,
        combination_maxdim,
        breakdown_tolerance,
    )
    return (
        krylovdim = krylovdim,
        action_cutoff = action_cutoff,
        action_maxdim = action_maxdim,
        orthogonalization_cutoff = orthogonalization_cutoff,
        orthogonalization_maxdim = orthogonalization_maxdim,
        combination_cutoff = combination_cutoff,
        combination_maxdim = combination_maxdim,
        breakdown_tolerance = breakdown_tolerance,
    )
end

"""Advance one state with `exp(-im * midpoint_hamiltonian * dt)` globally.

A full, explicitly compressed Arnoldi basis is constructed and its projected
Hessenberg matrix is exponentiated with dense linear algebra. The result is a
compressed linear combination of that basis and is intentionally *not*
renormalized. Diagnostics expose the projected residual, basis defects, norm
drift, and all compression controls.

For `dt == 0`, this returns a copy without constructing an Arnoldi basis. Its
diagnostics report zero accepted dimension and residual.
"""
function midpoint_global_krylov_step(
        state::MPS,
        midpoint_hamiltonian,
        dt::Real;
        krylovdim::Integer,
        action_cutoff::Real,
        action_maxdim::Integer,
        orthogonalization_cutoff::Real,
        orthogonalization_maxdim::Integer,
        combination_cutoff::Real,
        combination_maxdim::Integer,
        breakdown_tolerance::Real,
    )
    _validate_global_krylov_controls(
        dt,
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        combination_cutoff,
        combination_maxdim,
        breakdown_tolerance,
    )
    controls = _global_krylov_controls(
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        combination_cutoff,
        combination_maxdim,
        breakdown_tolerance,
    )
    input_norm = mps_norm(state)

    if iszero(dt)
        return (
            state = copy(state),
            diagnostics = (
                accepted_dimension = 0,
                breakdown = false,
                hessenberg = zeros(ComplexF64, 0, 0),
                projected_hermiticity_defect = 0.0,
                first_pass_defects = zeros(ComplexF64, 0, 0),
                second_pass_defects = zeros(ComplexF64, 0, 0),
                projected_residual = 0.0,
                input_norm,
                output_norm = input_norm,
                norm_drift = 0.0,
                controls,
            ),
        )
    end

    arnoldi = global_mps_arnoldi_basis(
        state,
        midpoint_hamiltonian;
        krylovdim,
        action_cutoff,
        action_maxdim,
        orthogonalization_cutoff,
        orthogonalization_maxdim,
        breakdown_tolerance,
    )
    dimension = arnoldi.accepted_dimension
    projected_hamiltonian = arnoldi.hessenberg[1:dimension, 1:dimension]
    initial_coordinate = zeros(ComplexF64, dimension)
    initial_coordinate[1] = 1.0
    coefficients = exp(-im * dt * projected_hamiltonian) * initial_coordinate
    evolved = mps_linear_combination(
        arnoldi.basis,
        coefficients;
        cutoff = combination_cutoff,
        maxdim = combination_maxdim,
    )
    residual_coefficient = arnoldi.hessenberg[dimension + 1, dimension]
    projected_residual = abs(residual_coefficient * coefficients[end])
    output_norm = mps_norm(evolved)

    return (
        state = evolved,
        diagnostics = (
            accepted_dimension = dimension,
            breakdown = arnoldi.breakdown,
            hessenberg = arnoldi.hessenberg,
            projected_hermiticity_defect = norm(projected_hamiltonian - projected_hamiltonian'),
            first_pass_defects = arnoldi.first_pass_defects,
            second_pass_defects = arnoldi.second_pass_defects,
            projected_residual,
            input_norm,
            output_norm,
            norm_drift = output_norm - input_norm,
            controls,
        ),
    )
end
