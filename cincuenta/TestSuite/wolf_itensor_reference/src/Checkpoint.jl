"""Crash-safe HDF5 checkpoints for streamed Wolf-reference trajectories."""

using HDF5
using ITensorMPS

const WOLF_CHECKPOINT_SCHEMA_VERSION = 1

"""Atomically save both spin-component MPS states and restart provenance."""
function write_wolf_checkpoint(
        path::AbstractString;
        up_state::MPS,
        down_state::MPS,
        completed_step::Integer,
        maximum_projected_residual::Real,
        signature::AbstractString,
        endpoints::AbstractVector{<:Real},
        couplings::AbstractMatrix,
    )
    completed_step >= 0 || throw(ArgumentError("completed_step must be nonnegative"))
    completed_step < length(endpoints) ||
        throw(ArgumentError("completed_step must identify an endpoint"))
    isfinite(maximum_projected_residual) && maximum_projected_residual >= 0 ||
        throw(ArgumentError("maximum_projected_residual must be finite and nonnegative"))
    size(couplings, 1) == 2 * length(endpoints) - 1 || throw(DimensionMismatch(
        "couplings must contain interleaved endpoint and midpoint rows"
    ))
    isempty(signature) && throw(ArgumentError("checkpoint signature must not be empty"))

    directory = dirname(path)
    mkpath(directory)
    temporary_path = path * ".tmp.$(getpid())"
    ispath(temporary_path) && rm(temporary_path; force = true)
    try
        h5open(temporary_path, "w") do file
            attributes(file)["schema_version"] = WOLF_CHECKPOINT_SCHEMA_VERSION
            attributes(file)["completed_step"] = Int(completed_step)
            attributes(file)["maximum_projected_residual"] =
                Float64(maximum_projected_residual)
            attributes(file)["signature"] = String(signature)
            file["endpoints"] = Float64.(endpoints)
            file["couplings"] = ComplexF64.(couplings)
            write(file, "up_state", up_state)
            write(file, "down_state", down_state)
        end
        mv(temporary_path, path; force = true)
    finally
        ispath(temporary_path) && rm(temporary_path; force = true)
    end
    return path
end

"""Load a checkpoint, requiring the current schema and complete payload."""
function read_wolf_checkpoint(path::AbstractString)
    isfile(path) || throw(ArgumentError("checkpoint does not exist: $path"))
    return h5open(path, "r") do file
        attrs = attributes(file)
        haskey(attrs, "schema_version") ||
            throw(ArgumentError("checkpoint has no schema version"))
        read(attrs["schema_version"]) == WOLF_CHECKPOINT_SCHEMA_VERSION ||
            throw(ArgumentError("unsupported checkpoint schema version"))
        for name in ("completed_step", "maximum_projected_residual", "signature")
            haskey(attrs, name) || throw(ArgumentError("checkpoint lacks attribute $name"))
        end
        for name in ("endpoints", "couplings", "up_state", "down_state")
            haskey(file, name) || throw(ArgumentError("checkpoint lacks dataset $name"))
        end
        return (
            completed_step = Int(read(attrs["completed_step"])),
            maximum_projected_residual =
                Float64(read(attrs["maximum_projected_residual"])),
            signature = String(read(attrs["signature"])),
            endpoints = read(file, "endpoints"),
            couplings = read(file, "couplings"),
            up_state = read(file, "up_state", MPS),
            down_state = read(file, "down_state", MPS),
        )
    end
end
