"""Exact one-particle reference for the noninteracting factorized SIAM.

These functions use the same data-only `model` specification as the MPO
builder, but construct the single-spin one-particle Hamiltonian directly. They
are an independent validation path for `interaction == 0`; they are not an
MPS algorithm and do not read configuration files.
"""

using LinearAlgebra

"""Return the one-spin Hamiltonian matrix for one explicit model time index.

The matrix convention is `H = sum(h[i,j] * cdag_i * c_j)`. It is valid only
for a zero-interaction model. Its row and column order is exactly the model's
physical MPS order, including the central impurity.
"""
function free_one_particle_hamiltonian(model, time_index::Integer)
    iszero(model.interaction) ||
        throw(ArgumentError("one-particle propagation requires interaction == 0"))

    site_count = length(model.site_order)
    impurity = model.impurity_position
    couplings = couplings_at(model, time_index)
    hamiltonian = zeros(ComplexF64, site_count, site_count)
    hamiltonian[impurity, impurity] = -model.chemical_potential

    for bath_label in 1:model.bath.bath_sites
        bath = bath_position(model, bath_label)
        coupling = couplings[bath_label]
        hamiltonian[bath, bath] = model.bath.bath_potentials[bath_label]
        hamiltonian[impurity, bath] = coupling
        hamiltonian[bath, impurity] = conj(coupling)
    end
    return hamiltonian
end

"""Return the initial one-spin density matrix for one atomic spin component.

The convention is `rho[i,j] = <cdag_j c_i>`, so the diagonal is the occupation
of the physical site at that MPS position. The state labels are obtained from
`model`, and `spin` must be `:Up` or `:Dn`.
"""
function initial_one_particle_density(model, impurity_state::Symbol, spin::Symbol)
    spin in (:Up, :Dn) || throw(ArgumentError("spin must be :Up or :Dn"))
    labels = initial_product_labels(model, impurity_state)
    occupations = map(labels) do label
        if label == :Emp
            0.0
        elseif label == :UpDn
            1.0
        elseif label == spin
            1.0
        elseif label in (:Up, :Dn)
            0.0
        else
            throw(ArgumentError("unsupported electron-state label: $label"))
        end
    end
    return Matrix(Diagonal(occupations))
end

"""Propagate one density matrix through an explicit constant midpoint matrix.

For `rho[i,j] = <cdag_j c_i>` and `U = exp(-im * hamiltonian * dt)`, the
exact one-particle update is `rho_next = U * rho * U†`. The caller supplies
both the midpoint matrix and `dt`; no time-grid convention is hidden here.
"""
function free_midpoint_step(
        density::AbstractMatrix,
        hamiltonian::AbstractMatrix,
        dt::Real;
        atol::Real = 1e-12,
        rtol::Real = 1e-10,
    )
    isfinite(dt) && dt >= 0 || throw(ArgumentError("dt must be finite and nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))
    size(hamiltonian, 1) == size(hamiltonian, 2) ||
        throw(ArgumentError("one-particle Hamiltonian must be square"))
    size(density) == size(hamiltonian) ||
        throw(ArgumentError("density and Hamiltonian dimensions must agree"))
    isapprox(hamiltonian, hamiltonian'; atol, rtol) ||
        throw(ArgumentError("one-particle Hamiltonian must be Hermitian"))
    isapprox(density, density'; atol, rtol) ||
        throw(ArgumentError("one-particle density must be Hermitian"))

    propagator = exp(-im * dt * hamiltonian)
    return propagator * density * propagator'
end

"""Propagate a density through caller-supplied midpoint matrices.

The returned `densities` include a copy of the initial density followed by one
exact one-particle result per endpoint interval. It does not construct an MPS
or retain mutable state.
"""
function run_free_midpoint_steps(
        initial_density::AbstractMatrix,
        endpoint_times::AbstractVector{<:Real},
        midpoint_hamiltonians::AbstractVector{<:AbstractMatrix};
        kwargs...,
    )
    step_count = length(endpoint_times) - 1
    step_count >= 1 || throw(ArgumentError("at least two endpoint times are required"))
    length(midpoint_hamiltonians) == step_count ||
        throw(ArgumentError("one midpoint Hamiltonian is required per interval"))
    all(isfinite, endpoint_times) || throw(ArgumentError("endpoint times must be finite"))
    all(diff(endpoint_times) .> 0) ||
        throw(ArgumentError("endpoint times must be strictly increasing"))

    density = complex.(initial_density)
    densities = Matrix{ComplexF64}[copy(density)]
    for step in eachindex(midpoint_hamiltonians)
        density = free_midpoint_step(
            density,
            midpoint_hamiltonians[step],
            endpoint_times[step + 1] - endpoint_times[step];
            kwargs...,
        )
        push!(densities, density)
    end
    return (times = Float64.(endpoint_times), densities)
end

"""Return local and conserved one-particle quantities from an explicit density."""
function one_particle_density_diagnostics(density::AbstractMatrix, impurity_position::Integer)
    size(density, 1) == size(density, 2) ||
        throw(ArgumentError("density must be square"))
    1 <= impurity_position <= size(density, 1) ||
        throw(ArgumentError("impurity_position must index the supplied density"))
    return (
        total_particle_number = real(tr(density)),
        impurity_occupation = real(density[impurity_position, impurity_position]),
        particle_number_imaginary_part = imag(tr(density)),
        impurity_occupation_imaginary_part = imag(density[impurity_position, impurity_position]),
    )
end
