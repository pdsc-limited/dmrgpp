using TOML

"""Load a declarative Wolf benchmark configuration from `path`."""
load_benchmark_config(path::AbstractString) = TOML.parsefile(path)

"""Return contract violations in a parsed Wolf benchmark configuration.

This function is deliberately side-effect free: callers decide whether a
nonempty error list is fatal for their use case.
"""
function validate_benchmark_config(config::AbstractDict)
    errors = String[]
    benchmark = get(config, "benchmark", nothing)
    model = get(config, "model", nothing)
    bath = get(config, "bath", nothing)
    time_grid = get(config, "time_grid", nothing)
    initial_state = get(config, "initial_state", nothing)

    for (name, value) in (
        ("benchmark", benchmark),
        ("model", model),
        ("bath", bath),
        ("time_grid", time_grid),
        ("initial_state", initial_state),
    )
        value isa AbstractDict || push!(errors, "missing [$name] table")
    end
    isempty(errors) || return errors

    get(benchmark, "schema_version", nothing) == 1 ||
        push!(errors, "benchmark.schema_version must be 1")
    get(model, "beta", nothing) == 1.0 || push!(errors, "model.beta must be 1.0")
    get(model, "U", nothing) in (4.0, 10.0) ||
        push!(errors, "model.U must be 4.0 or 10.0")
    get(bath, "representation", nothing) == "time_grid_factorization" ||
        push!(errors, "bath.representation must be time_grid_factorization")
    get(bath, "initial_occupied_sector", nothing) == "lesser" ||
        push!(errors, "bath.initial_occupied_sector must be lesser")
    get(bath, "initial_empty_sector", nothing) == "greater" ||
        push!(errors, "bath.initial_empty_sector must be greater")

    discretizations = get(bath, "discretizations", nothing)
    discretizations isa AbstractVector || begin
        push!(errors, "bath.discretizations must be an array")
        return errors
    end
    isempty(discretizations) && push!(errors, "bath.discretizations must not be empty")
    for entry in discretizations
        entry isa AbstractDict || begin
            push!(errors, "bath discretization must be a table")
            continue
        end
        lb = get(entry, "Lb", nothing)
        lesser_rank = get(entry, "lesser_rank", nothing)
        greater_rank = get(entry, "greater_rank", nothing)
        lb isa Integer && lb >= 10 ||
            push!(errors, "Lb must be an integer at least 10")
        lb isa Integer && iseven(lb) || push!(errors, "Lb must be even")
        if lb isa Integer && iseven(lb)
            lesser_rank == lb ÷ 2 || push!(errors, "lesser_rank must equal Lb / 2")
            greater_rank == lb ÷ 2 || push!(errors, "greater_rank must equal Lb / 2")
        end
    end

    get(time_grid, "status", nothing) in ("provisional", "resolved") ||
        push!(errors, "time_grid.status must be provisional or resolved")
    get(initial_state, "status", nothing) in ("unresolved", "resolved") ||
        push!(errors, "initial_state.status must be unresolved or resolved")
    return errors
end

"""Whether a configuration has resolved the two historical blockers for a run."""
function benchmark_is_runnable(config::AbstractDict)
    return get(get(config, "time_grid", Dict()), "status", nothing) == "resolved" &&
           get(get(config, "initial_state", Dict()), "status", nothing) == "resolved" &&
           isempty(validate_benchmark_config(config))
end
