"""Explicit endpoint records and spin-component averaging for global Krylov runs.

This module is deliberately small: callers supply all states, endpoint and
midpoint MPOs, times, and Krylov controls. It does not construct a bath,
choose a grid, write output, or retain a hidden trajectory.
"""

using ITensorMPS: MPS

"""Measure one endpoint state at one explicit finite time."""
function global_krylov_endpoint_record(state::MPS, time::Real, hamiltonian, impurity_position::Integer)
    isfinite(time) || throw(ArgumentError("endpoint time must be finite"))
    return (time = float(time), diagnostics = state_diagnostics(state, hamiltonian, impurity_position))
end

"""Evolve one atomic component through caller-supplied midpoint MPOs.

The returned records include the initial endpoint. Each endpoint MPO is used
only for measurement, while each midpoint MPO is used only for its matching
Krylov step. All numerical controls are forwarded explicitly to
`midpoint_global_krylov_step`.
"""
function run_global_krylov_component(
        initial_state::MPS,
        endpoint_times::AbstractVector{<:Real},
        midpoint_hamiltonians::AbstractVector,
        endpoint_hamiltonians::AbstractVector,
        impurity_position::Integer;
        kwargs...,
    )
    step_count = length(endpoint_times) - 1
    step_count >= 1 || throw(ArgumentError("at least two endpoint times are required"))
    length(midpoint_hamiltonians) == step_count ||
        throw(ArgumentError("one midpoint Hamiltonian is required per interval"))
    length(endpoint_hamiltonians) == length(endpoint_times) ||
        throw(ArgumentError("one endpoint Hamiltonian is required per endpoint"))
    all(isfinite, endpoint_times) || throw(ArgumentError("endpoint times must be finite"))
    all(diff(endpoint_times) .> 0) ||
        throw(ArgumentError("endpoint times must be strictly increasing"))

    states = MPS[copy(initial_state)]
    records = [
        global_krylov_endpoint_record(
            states[1], endpoint_times[1], endpoint_hamiltonians[1], impurity_position
        ),
    ]
    step_diagnostics = NamedTuple[]
    for step in eachindex(midpoint_hamiltonians)
        evolved = midpoint_global_krylov_step(
            states[end],
            midpoint_hamiltonians[step],
            endpoint_times[step + 1] - endpoint_times[step];
            kwargs...,
        )
        push!(states, evolved.state)
        push!(step_diagnostics, evolved.diagnostics)
        push!(
            records,
            global_krylov_endpoint_record(
                evolved.state,
                endpoint_times[step + 1],
                endpoint_hamiltonians[step + 1],
                impurity_position,
            ),
        )
    end
    return (states = states, records = records, step_diagnostics = step_diagnostics)
end

"""Average scalar endpoint diagnostics from atomic `:Up` and `:Dn` components.

Both records must represent the same endpoint. Scalar fields are averaged;
`max_link_dimension` is the maximum of the two component states because it is
an allocation diagnostic, not an observable.
"""
function average_spin_component_record(up_record, down_record)
    up_record.time == down_record.time ||
        throw(ArgumentError("spin-component records must have identical times"))
    up = up_record.diagnostics
    down = down_record.diagnostics
    return (
        time = up_record.time,
        diagnostics = (
            norm = spin_average(up.norm, down.norm),
            total_particle_number = spin_average(
                up.total_particle_number, down.total_particle_number
            ),
            spin_projection = spin_average(up.spin_projection, down.spin_projection),
            energy = spin_average(up.energy, down.energy),
            energy_imaginary_part = spin_average(
                up.energy_imaginary_part, down.energy_imaginary_part
            ),
            impurity_nup = spin_average(up.impurity_nup, down.impurity_nup),
            impurity_ndn = spin_average(up.impurity_ndn, down.impurity_ndn),
            impurity_double_occupancy = spin_average(
                up.impurity_double_occupancy, down.impurity_double_occupancy
            ),
            max_link_dimension = max(up.max_link_dimension, down.max_link_dimension),
        ),
    )
end

"""Average matching vectors of atomic-spin endpoint records."""
function average_spin_component_records(up_records::AbstractVector, down_records::AbstractVector)
    length(up_records) == length(down_records) ||
        throw(ArgumentError("spin-component record counts must agree"))
    isempty(up_records) && throw(ArgumentError("at least one endpoint record is required"))
    return [
        average_spin_component_record(up_records[index], down_records[index])
        for index in eachindex(up_records)
    ]
end
