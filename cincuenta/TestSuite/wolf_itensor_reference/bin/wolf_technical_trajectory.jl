#!/usr/bin/env julia

"""Run one explicitly parameterized, bounded technical Wolf-reference trajectory.

This program is intentionally not a benchmark runner: the caller must supply
`--dt`, `--tmax`, and `--Lb`, and generated metadata labels the output as a
technical validation result. It performs no self-consistency loop.
"""

const ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(ROOT, "src", "WolfITensorReference.jl"))
using .WolfITensorReference
using ITensors
using ITensorMPS: siteinds
using LinearAlgebra

function usage(io = stdout)
    println(io, "usage: wolf_technical_trajectory.jl --config FILE --out DIR --Lb N --dt X --tmax X [options]")
    println(io, "options: --krylovdim N --maxdim N --cutoff X --intervals N --progress-every N")
    println(io, "         --threaded-blocksparse BOOL --parallel-components BOOL --gc-every N")
    println(io, "         --checkpoint-every N --restart FILE --stop-after-step N")
end

function parse_options(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        argument in ("--help", "-h") && return nothing
        startswith(argument, "--") || throw(ArgumentError("unexpected argument: $argument"))
        index < length(arguments) || throw(ArgumentError("missing value for $argument"))
        haskey(options, argument) && throw(ArgumentError("duplicate option: $argument"))
        options[argument] = arguments[index + 1]
        index += 2
    end
    required = ("--config", "--out", "--Lb", "--dt", "--tmax")
    all(option -> haskey(options, option), required) ||
        throw(ArgumentError("required options: $(join(required, ", "))"))
    return options
end

parse_integer(options, name) = try
    parse(Int, options[name])
catch
    throw(ArgumentError("$name must be an integer"))
end
parse_real(options, name) = try
    parse(Float64, options[name])
catch
    throw(ArgumentError("$name must be a floating-point number"))
end
function parse_boolean(options, name)
    value = lowercase(options[name])
    value in ("true", "yes", "1") && return true
    value in ("false", "no", "0") && return false
    throw(ArgumentError("$name must be true or false"))
end

function records_from_table(table, count)
    count <= length(table.time) || throw(ArgumentError(
        "trajectory CSV is shorter than the checkpoint prefix"
    ))
    return [
        (
            time = table.time[index],
            diagnostics = NamedTuple{TRAJECTORY_COLUMNS[2:end]}(Tuple(
                column === :max_link_dimension ?
                    round(Int, getproperty(table, column)[index]) :
                    getproperty(table, column)[index]
                for column in TRAJECTORY_COLUMNS[2:end]
            )),
        ) for index in 1:count
    ]
end

function selected_rank(config, bath_sites)
    entries = config["bath"]["discretizations"]
    selected = findfirst(entry -> entry["Lb"] == bath_sites, entries)
    selected === nothing && throw(ArgumentError("Lb=$bath_sites is not listed in the config"))
    entry = entries[selected]
    return (lesser = entry["lesser_rank"], greater = entry["greater_rank"])
end

function uniform_endpoints(dt, tmax)
    isfinite(dt) && dt > 0 || throw(ArgumentError("--dt must be positive and finite"))
    isfinite(tmax) && tmax > 0 || throw(ArgumentError("--tmax must be positive and finite"))
    steps = round(Int, tmax / dt)
    steps >= 1 && isapprox(steps * dt, tmax; atol = 1e-12, rtol = 1e-12) ||
        throw(ArgumentError("--tmax must be an integer multiple of --dt"))
    return collect(0:steps) .* dt
end

function main(arguments)
    options = parse_options(arguments)
    options === nothing && return usage()
    config_path = normpath(options["--config"])
    config = load_benchmark_config(config_path)
    errors = validate_benchmark_config(config)
    isempty(errors) || throw(ArgumentError(join(errors, "; ")))
    bath_sites = parse_integer(options, "--Lb")
    bath_sites >= 10 && iseven(bath_sites) ||
        throw(ArgumentError("--Lb must be even and at least 10"))
    ranks = selected_rank(config, bath_sites)
    dt = parse_real(options, "--dt")
    tmax = parse_real(options, "--tmax")
    endpoints = uniform_endpoints(dt, tmax)
    krylovdim = haskey(options, "--krylovdim") ? parse_integer(options, "--krylovdim") : 7
    maxdim = haskey(options, "--maxdim") ? parse_integer(options, "--maxdim") : 1000
    cutoff = haskey(options, "--cutoff") ? parse_real(options, "--cutoff") : 0.0
    intervals = haskey(options, "--intervals") ? parse_integer(options, "--intervals") : 512
    intervals > 0 && iseven(intervals) || throw(ArgumentError("--intervals must be positive and even"))
    progress_every = haskey(options, "--progress-every") ? parse_integer(options, "--progress-every") : 10
    progress_every > 0 || throw(ArgumentError("--progress-every must be positive"))
    gc_every = haskey(options, "--gc-every") ? parse_integer(options, "--gc-every") : 0
    gc_every >= 0 || throw(ArgumentError("--gc-every must be nonnegative"))
    threaded_blocksparse = haskey(options, "--threaded-blocksparse") ?
        parse_boolean(options, "--threaded-blocksparse") : false
    parallel_components = haskey(options, "--parallel-components") ?
        parse_boolean(options, "--parallel-components") : false
    checkpoint_every = haskey(options, "--checkpoint-every") ?
        parse_integer(options, "--checkpoint-every") : 0
    checkpoint_every >= 0 || throw(ArgumentError("--checkpoint-every must be nonnegative"))
    stop_after_step = haskey(options, "--stop-after-step") ?
        parse_integer(options, "--stop-after-step") : length(endpoints) - 1
    0 <= stop_after_step <= length(endpoints) - 1 || throw(ArgumentError(
        "--stop-after-step must lie between zero and the number of timesteps"
    ))
    restart_path = get(options, "--restart", "")
    threaded_blocksparse && parallel_components && throw(ArgumentError(
        "select either block-sparse threading or parallel components, not both"
    ))
    parallel_components && Threads.nthreads() < 2 && throw(ArgumentError(
        "--parallel-components requires Julia to be started with at least two threads"
    ))
    if threaded_blocksparse
        Threads.nthreads() > 1 || throw(ArgumentError(
            "--threaded-blocksparse requires Julia to be started with -t N, N > 1"
        ))
        BLAS.set_num_threads(1)
        ITensors.Strided.disable_threads()
        ITensors.enable_threaded_blocksparse()
    end
    println(
        "threads=$(Threads.nthreads()) blas_threads=$(BLAS.get_num_threads()) " *
        "threaded_blocksparse=$(ITensors.using_threaded_blocksparse()) " *
        "parallel_components=$parallel_components gc_every=$gc_every"
    )

    output_directory = normpath(options["--out"])
    mkpath(output_directory)
    trajectory_path = joinpath(output_directory, "trajectory.csv")
    metadata_path = joinpath(output_directory, "metadata.toml")
    checkpoint_path = joinpath(output_directory, "checkpoint.h5")

    grid = midpoint_factorization_grid(endpoints)
    t1 = config["time_grid"]["ramp_duration"]
    lesser_kernel = -im * hybridization_matrix(
        endpoints; component = :lesser, t1, intervals
    )
    greater_kernel = im * hybridization_matrix(
        endpoints; component = :greater, t1, intervals
    )
    ranks.lesser == ranks.greater || throw(ArgumentError(
        "particle-hole-symmetric baths require equal lesser and greater ranks"
    ))
    lesser = factorize_causal_cholesky(lesser_kernel; rank = ranks.lesser, atol = 1e-11)
    # At particle-hole symmetry i*Lambda^> = conj(-i*Lambda^<). Wolf notes
    # that only one decomposition is needed: initially empty bath couplings
    # are the complex conjugates of the occupied-bath couplings. Factoring
    # the two nearly singular kernels independently can break this symmetry.
    greater_factor = conj.(lesser.factor)
    greater_reconstruction = greater_factor * greater_factor'
    greater_reconstruction_error = norm(greater_kernel - greater_reconstruction) /
        max(norm(greater_kernel), 1.0)
    greater_maximum_absolute_error = maximum(
        abs, greater_kernel - greater_reconstruction; init = 0.0
    )
    lesser_rows = causal_midpoint_factor_rows(endpoints, lesser.factor)
    greater_rows = conj.(lesser_rows)
    bath = factorized_bath_spec(lesser_rows, greater_rows)
    model = siam_model_spec(config["model"]["U"], bath; chemical_potential = config["model"]["chemical_potential"])
    controls = (
        krylovdim,
        action_cutoff = cutoff,
        action_maxdim = maxdim,
        orthogonalization_cutoff = cutoff,
        orthogonalization_maxdim = maxdim,
        combination_cutoff = cutoff,
        combination_maxdim = maxdim,
        breakdown_tolerance = 1e-12,
    )

    signature = join((
        "Lb=$bath_sites",
        "U=$(model.interaction)",
        "mu=$(model.chemical_potential)",
        "dt=$dt",
        "tmax=$tmax",
        "t1=$t1",
        "intervals=$intervals",
        "lesser_rank=$(ranks.lesser)",
        "greater_rank=$(ranks.greater)",
        "krylovdim=$krylovdim",
        "cutoff=$cutoff",
        "maxdim=$maxdim",
    ), ';')

    # Stream the two historical spin components together. Long trajectories
    # retain only the current pair of MPS states, rather than every state and
    # every endpoint/midpoint MPO.
    if isempty(restart_path)
        sites = electron_sites(model)
        up_state = initial_product_mps(model, sites, :Up)
        down_state = initial_product_mps(model, sites, :Dn)
        initial_hamiltonian = siam_mpo(model, sites, grid.endpoint_indices[1])
        records = [
            average_spin_component_record(
                global_krylov_endpoint_record(
                    up_state, endpoints[1], initial_hamiltonian, model.impurity_position
                ),
                global_krylov_endpoint_record(
                    down_state, endpoints[1], initial_hamiltonian, model.impurity_position
                ),
            ),
        ]
        maximum_projected_residual = 0.0
        completed_step = 0
    else
        restored = read_wolf_checkpoint(normpath(restart_path))
        restored.signature == signature || throw(ArgumentError(
            "checkpoint numerical controls do not match this run"
        ))
        restored.endpoints == endpoints || throw(ArgumentError(
            "checkpoint endpoint grid does not match this run"
        ))
        restored.couplings == bath.couplings || throw(ArgumentError(
            "checkpoint causal bath couplings do not match this run"
        ))
        siteinds(restored.up_state) == siteinds(restored.down_state) ||
            throw(ArgumentError("checkpoint spin components use different site indices"))
        completed_step = restored.completed_step
        completed_step <= stop_after_step || throw(ArgumentError(
            "--stop-after-step precedes the restored checkpoint"
        ))
        table = read_trajectory_csv(trajectory_path)
        records = records_from_table(table, completed_step + 1)
        records[end].time == endpoints[completed_step + 1] || throw(ArgumentError(
            "trajectory CSV endpoint does not match the checkpoint"
        ))
        up_state = restored.up_state
        down_state = restored.down_state
        sites = siteinds(up_state)
        maximum_projected_residual = restored.maximum_projected_residual
        println("restarted from step $completed_step at t=$(endpoints[completed_step + 1])")
    end

    final_step = stop_after_step
    for step in (completed_step + 1):final_step
        midpoint_hamiltonian = siam_mpo(model, sites, grid.midpoint_indices[step])
        dt_step = endpoints[step + 1] - endpoints[step]
        if parallel_components
            up_task = Threads.@spawn midpoint_global_krylov_step(
                up_state, midpoint_hamiltonian, dt_step; controls...
            )
            down_task = Threads.@spawn midpoint_global_krylov_step(
                down_state, midpoint_hamiltonian, dt_step; controls...
            )
            up_evolved = fetch(up_task)
            down_evolved = fetch(down_task)
        else
            up_evolved = midpoint_global_krylov_step(
                up_state, midpoint_hamiltonian, dt_step; controls...
            )
            down_evolved = midpoint_global_krylov_step(
                down_state, midpoint_hamiltonian, dt_step; controls...
            )
        end
        up_state = up_evolved.state
        down_state = down_evolved.state
        maximum_projected_residual = max(
            maximum_projected_residual,
            up_evolved.diagnostics.projected_residual,
            down_evolved.diagnostics.projected_residual,
        )

        endpoint_hamiltonian = siam_mpo(model, sites, grid.endpoint_indices[step + 1])
        push!(
            records,
            average_spin_component_record(
                global_krylov_endpoint_record(
                    up_state,
                    endpoints[step + 1],
                    endpoint_hamiltonian,
                    model.impurity_position,
                ),
                global_krylov_endpoint_record(
                    down_state,
                    endpoints[step + 1],
                    endpoint_hamiltonian,
                    model.impurity_position,
                ),
            ),
        )
        if step % progress_every == 0 || step == final_step
            record = records[end]
            println(
                "step $step/$(length(grid.midpoint_indices)) t=$(record.time) " *
                "d=$(record.diagnostics.impurity_double_occupancy) " *
                "maxdim=$(record.diagnostics.max_link_dimension) " *
                "residual=$maximum_projected_residual",
            )
            write_trajectory_csv(trajectory_path, records)
        end
        if checkpoint_every > 0 && step % checkpoint_every == 0
            write_trajectory_csv(trajectory_path, records)
            write_wolf_checkpoint(
                checkpoint_path;
                up_state,
                down_state,
                completed_step = step,
                maximum_projected_residual,
                signature,
                endpoints,
                couplings = bath.couplings,
            )
            println("checkpointed step $step at $checkpoint_path")
        end

        # Krylov construction allocates several temporary MPS vectors and
        # compressed direct sums. Release those between streamed timesteps so
        # long runs do not retain unreachable tensor storage in Julia's heap.
        midpoint_hamiltonian = nothing
        endpoint_hamiltonian = nothing
        up_evolved = nothing
        down_evolved = nothing
        gc_every > 0 && step % gc_every == 0 && GC.gc()
    end

    write_trajectory_csv(trajectory_path, records)
    final_checkpoint_already_written = checkpoint_every > 0 &&
        final_step > completed_step && final_step % checkpoint_every == 0
    if (final_step < length(grid.midpoint_indices) || checkpoint_every > 0) &&
            !final_checkpoint_already_written
        write_wolf_checkpoint(
            checkpoint_path;
            up_state,
            down_state,
            completed_step = final_step,
            maximum_projected_residual,
            signature,
            endpoints,
            couplings = bath.couplings,
        )
        println("checkpointed step $final_step at $checkpoint_path")
    end
    completion_status = final_step == length(grid.midpoint_indices) ?
        "completed" : "stopped_at_checkpoint"
    metadata = Dict(
        "schema_version" => TRAJECTORY_SCHEMA_VERSION,
        "program" => "wolf_technical_trajectory",
        "program_status" => "bounded technical trajectory; not a Wolf benchmark result",
        "completion_status" => completion_status,
        "config_path" => config_path,
        "run" => Dict("Lb" => bath_sites, "U" => model.interaction, "beta" => config["model"]["beta"], "dt" => dt, "tmax" => tmax, "ramp_duration" => t1, "endpoint_times" => endpoints),
        "factorization" => Dict("method" => "causal optimized low-rank Cholesky (GBEK Eqs. 56-63)", "particle_hole_pairing" => "greater couplings are the complex conjugates of lesser couplings", "midpoint_interpolation" => "natural cubic spline from the causal endpoint prefix", "lesser_rank" => lesser.rank, "greater_rank" => ranks.greater, "lesser_reconstruction_error" => lesser.reconstruction_error, "greater_reconstruction_error" => greater_reconstruction_error, "lesser_maximum_absolute_error" => lesser.maximum_absolute_error, "greater_maximum_absolute_error" => greater_maximum_absolute_error),
        "global_krylov" => Dict("krylovdim" => krylovdim, "cutoff" => cutoff, "maxdim" => maxdim, "maximum_projected_residual" => maximum_projected_residual),
        "execution" => Dict("julia_threads" => Threads.nthreads(), "blas_threads" => BLAS.get_num_threads(), "threaded_blocksparse" => ITensors.using_threaded_blocksparse(), "parallel_components" => parallel_components, "gc_every" => gc_every, "checkpoint_every" => checkpoint_every, "completed_step" => final_step),
        "initial_state" => "separate atomic Up and Dn components; scalar records averaged",
        "ramp_duration_status" => config["time_grid"]["ramp_duration_status"],
    )
    write_metadata_toml(metadata_path, metadata)
    println("wrote $trajectory_path")
    println("wrote $metadata_path")
end

try
    main(ARGS)
catch error
    println(stderr, "error: ", sprint(showerror, error))
    usage(stderr)
    exit(1)
end
