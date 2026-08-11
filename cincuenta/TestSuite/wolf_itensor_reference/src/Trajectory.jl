"""Bounded, caller-directed midpoint trajectory composition.

This module performs no factorization, grid construction, configuration I/O, or
file output. Callers supply every endpoint and Hamiltonian explicitly, so the
numerical trajectory convention remains visible at the call site.
"""

"""Measure one state at an explicit finite endpoint time."""
function endpoint_record(state, time::Real, endpoint_hamiltonian, impurity_position::Integer)
    isfinite(time) || throw(ArgumentError("endpoint time must be finite"))
    return (
        time = time,
        diagnostics = state_diagnostics(state, endpoint_hamiltonian, impurity_position),
    )
end

"""Evolve one explicit interval and measure the resulting endpoint state.

`midpoint_hamiltonian` is the caller's chosen Hamiltonian for propagation;
`endpoint_hamiltonian` is independently supplied for the reported endpoint
energy. Keeping them separate prevents this function from imposing a hidden
midpoint-to-endpoint convention.
"""
function midpoint_trajectory_step(
        state,
        start_time::Real,
        end_time::Real,
        midpoint_hamiltonian,
        endpoint_hamiltonian,
        impurity_position::Integer;
        kwargs...,
    )
    isfinite(start_time) || throw(ArgumentError("start time must be finite"))
    isfinite(end_time) || throw(ArgumentError("end time must be finite"))
    end_time > start_time || throw(ArgumentError("end time must exceed start time"))

    evolved = midpoint_tdvp_step(
        state,
        midpoint_hamiltonian,
        end_time - start_time;
        kwargs...,
    )
    return (state = evolved, record = endpoint_record(
        evolved,
        end_time,
        endpoint_hamiltonian,
        impurity_position,
    ))
end

"""Run a finite, explicitly supplied sequence of midpoint steps.

The returned `records` include the supplied initial state at `endpoint_times[1]`
and one record after each step. The function only owns local loop variables; it
retains no mutable trajectory state and performs no output.
"""
function run_midpoint_steps(
        initial_state,
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

    state = copy(initial_state)
    records = [endpoint_record(
        state,
        endpoint_times[1],
        endpoint_hamiltonians[1],
        impurity_position,
    )]
    for step in 1:step_count
        result = midpoint_trajectory_step(
            state,
            endpoint_times[step],
            endpoint_times[step + 1],
            midpoint_hamiltonians[step],
            endpoint_hamiltonians[step + 1],
            impurity_position;
            kwargs...,
        )
        state = result.state
        push!(records, result.record)
    end
    return (state = state, times = copy(endpoint_times), records = records)
end
