#!/usr/bin/env julia

"""Run a bounded end-to-end integration test of the Wolf reference plumbing.

This program is intentionally *not* a Wolf benchmark calculation. It uses the
explicit `Lb=10` contract entry, a short configurable number of TDVP steps,
and the provisional ramp duration recorded in the selected configuration. It
exists to verify configuration loading, kernel factorization, MPO construction,
propagation, measurements, and versioned output serialization together.

Usage:

    julia --project=. bin/wolf_integration_test.jl \
      --config configs/wolf_u4.toml --out output/integration-u4

Optional numerical controls are `--dt`, `--steps`, `--cutoff`, `--maxdim`,
and `--force`. The output directory is always explicit.
"""

using TOML
using ITensors
using ITensorMPS

const PROJECT_ROOT = normpath(joinpath(@__DIR__, ".."))
include(joinpath(PROJECT_ROOT, "src", "WolfITensorReference.jl"))
using .WolfITensorReference

const DEFAULT_CONTROLS = (
    dt = 0.02,
    steps = 2,
    cutoff = 1.0e-12,
    maxdim = 100,
    quadrature_intervals = 512,
)

function usage(io::IO = stderr)
    println(io, "usage: julia --project=. bin/wolf_integration_test.jl --config FILE --out DIRECTORY [--dt FLOAT] [--steps INT] [--cutoff FLOAT] [--maxdim INT] [--force]")
end

function parse_arguments(arguments)
    options = Dict{String,String}()
    index = 1
    while index <= length(arguments)
        argument = arguments[index]
        if argument == "--force"
            options[argument] = "true"
            index += 1
        elseif argument in ("--config", "--out", "--dt", "--steps", "--cutoff", "--maxdim")
            index < length(arguments) || throw(ArgumentError("$argument requires a value"))
            haskey(options, argument) && throw(ArgumentError("$argument was supplied more than once"))
            options[argument] = arguments[index + 1]
            index += 2
        else
            throw(ArgumentError("unknown argument: $argument"))
        end
    end
    for required in ("--config", "--out")
        haskey(options, required) || throw(ArgumentError("$required is required"))
    end

    controls = (
        dt = parse(Float64, get(options, "--dt", string(DEFAULT_CONTROLS.dt))),
        steps = parse(Int, get(options, "--steps", string(DEFAULT_CONTROLS.steps))),
        cutoff = parse(Float64, get(options, "--cutoff", string(DEFAULT_CONTROLS.cutoff))),
        maxdim = parse(Int, get(options, "--maxdim", string(DEFAULT_CONTROLS.maxdim))),
        quadrature_intervals = DEFAULT_CONTROLS.quadrature_intervals,
    )
    isfinite(controls.dt) && controls.dt > 0 || throw(ArgumentError("--dt must be finite and positive"))
    controls.steps >= 1 || throw(ArgumentError("--steps must be positive"))
    isfinite(controls.cutoff) && controls.cutoff >= 0 || throw(ArgumentError("--cutoff must be finite and nonnegative"))
    controls.maxdim >= 1 || throw(ArgumentError("--maxdim must be positive"))
    return (
        config_path = abspath(options["--config"]),
        output_directory = abspath(options["--out"]),
        force = haskey(options, "--force"),
        controls = controls,
    )
end

function lb10_ranks(config)
    entries = config["bath"]["discretizations"]
    entry = only(filter(entry -> entry["Lb"] == 10, entries))
    return (lesser = entry["lesser_rank"], greater = entry["greater_rank"])
end

function integration_metadata(config, arguments, factors, grid)
    return Dict(
        "schema_version" => TRAJECTORY_SCHEMA_VERSION,
        "program" => "wolf_integration_test",
        "program_status" => "technical integration test; not a Wolf benchmark result",
        "config_path" => arguments.config_path,
        "config" => config,
        "controls" => Dict(
            "dt" => arguments.controls.dt,
            "steps" => arguments.controls.steps,
            "cutoff" => arguments.controls.cutoff,
            "maxdim" => arguments.controls.maxdim,
            "quadrature_intervals" => arguments.controls.quadrature_intervals,
            "nsite" => 2,
        ),
        "bath" => Dict(
            "Lb" => 10,
            "lesser_rank" => factors.lesser.rank,
            "greater_rank" => factors.greater.rank,
            "lesser_discarded_spectral_weight" => factors.lesser.discarded_weight,
            "greater_discarded_spectral_weight" => factors.greater.discarded_weight,
            "lesser_relative_reconstruction_error" => factors.lesser.reconstruction_error,
            "greater_relative_reconstruction_error" => factors.greater.reconstruction_error,
        ),
        "factorization_grid" => Dict(
            "endpoint_times" => grid.endpoints,
            "midpoint_times" => grid.midpoints,
            "joint_points" => grid.points,
        ),
        "packages" => Dict(
            "julia" => string(VERSION),
            "ITensors" => string(pkgversion(ITensors)),
            "ITensorMPS" => string(pkgversion(ITensorMPS)),
        ),
        "completion_status" => "completed",
    )
end

function run_integration_test(arguments)
    config = load_benchmark_config(arguments.config_path)
    errors = validate_benchmark_config(config)
    isempty(errors) || throw(ArgumentError(join(errors, "; ")))
    isfile(joinpath(arguments.output_directory, "trajectory.csv")) && !arguments.force &&
        throw(ArgumentError("output already contains trajectory.csv; pass --force to overwrite"))

    t1 = config["time_grid"]["ramp_duration"]
    ranks = lb10_ranks(config)
    endpoint_times = collect(0:arguments.controls.dt:(arguments.controls.steps * arguments.controls.dt))
    grid = midpoint_factorization_grid(endpoint_times)

    lesser_kernel = -im * hybridization_matrix(
        grid.points;
        component = :lesser,
        t1,
        intervals = arguments.controls.quadrature_intervals,
    )
    greater_kernel = im * hybridization_matrix(
        grid.points;
        component = :greater,
        t1,
        intervals = arguments.controls.quadrature_intervals,
    )
    factors = (
        lesser = factorize_psd(lesser_kernel; rank = ranks.lesser, atol = 1e-11),
        greater = factorize_psd(greater_kernel; rank = ranks.greater, atol = 1e-11),
    )
    bath = factorized_bath_spec(factors.lesser.factor, factors.greater.factor)
    model = siam_model_spec(config["model"]["U"], bath;
        chemical_potential = config["model"]["chemical_potential"])
    sites = electron_sites(model)
    initial_state = initial_product_mps(model, sites, :Up)
    midpoint_hamiltonians = [siam_mpo(model, sites, index) for index in grid.midpoint_indices]
    endpoint_hamiltonians = [siam_mpo(model, sites, index) for index in grid.endpoint_indices]

    trajectory = run_midpoint_steps(
        initial_state,
        endpoint_times,
        midpoint_hamiltonians,
        endpoint_hamiltonians,
        model.impurity_position;
        cutoff = arguments.controls.cutoff,
        maxdim = arguments.controls.maxdim,
        nsite = 2,
        outputlevel = 0,
    )

    metadata = integration_metadata(config, arguments, factors, grid)
    mkpath(arguments.output_directory)
    write_trajectory_csv(joinpath(arguments.output_directory, "trajectory.csv"), trajectory.records)
    write_metadata_toml(joinpath(arguments.output_directory, "metadata.toml"), metadata)
    return trajectory
end

function main(arguments)
    try
        parsed = parse_arguments(arguments)
        trajectory = run_integration_test(parsed)
        println("technical integration test completed: $(length(trajectory.records)) endpoint records")
    catch error
        println(stderr, "error: ", sprint(showerror, error))
        usage()
        return 1
    end
    return 0
end

exit(main(ARGS))
