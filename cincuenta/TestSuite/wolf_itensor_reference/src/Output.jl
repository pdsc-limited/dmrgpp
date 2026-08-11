"""Small, versioned serializers for explicit trajectory records.

This module owns only data validation and file encoding. It neither constructs
states nor chooses numerical parameters. Callers provide records, metadata,
and output paths explicitly.
"""

using TOML

const TRAJECTORY_SCHEMA_VERSION = 1
const TRAJECTORY_COLUMNS = (
    :time,
    :norm,
    :total_particle_number,
    :spin_projection,
    :energy,
    :energy_imaginary_part,
    :impurity_nup,
    :impurity_ndn,
    :impurity_double_occupancy,
    :max_link_dimension,
)

"""Return schema errors for explicit endpoint records without writing files."""
function validate_trajectory_records(records)
    isempty(records) && return ["at least one trajectory record is required"]

    errors = String[]
    previous_time = nothing
    for (index, record) in enumerate(records)
        hasproperty(record, :time) || push!(errors, "record $index has no time")
        hasproperty(record, :diagnostics) || push!(errors, "record $index has no diagnostics")
        (!hasproperty(record, :time) || !hasproperty(record, :diagnostics)) && continue

        time = record.time
        time isa Real && isfinite(time) || push!(errors, "record $index time must be finite")
        if previous_time !== nothing && time isa Real && isfinite(time) && time <= previous_time
            push!(errors, "record times must be strictly increasing")
        end
        time isa Real && isfinite(time) && (previous_time = time)

        diagnostics = record.diagnostics
        for field in TRAJECTORY_COLUMNS[2:end]
            hasproperty(diagnostics, field) || push!(errors, "record $index diagnostics lack $field")
        end
        all(hasproperty(diagnostics, field) for field in TRAJECTORY_COLUMNS[2:end]) || continue

        for field in TRAJECTORY_COLUMNS[2:end-1]
            value = getproperty(diagnostics, field)
            value isa Real && isfinite(value) ||
                push!(errors, "record $index diagnostic $field must be finite")
        end
        max_link_dimension = diagnostics.max_link_dimension
        max_link_dimension isa Integer && max_link_dimension >= 1 ||
            push!(errors, "record $index max_link_dimension must be a positive integer")
    end
    return errors
end

"""Return a data-only, column-oriented table from endpoint records."""
function trajectory_table(records)
    errors = validate_trajectory_records(records)
    isempty(errors) || throw(ArgumentError(join(errors, "; ")))
    return NamedTuple{TRAJECTORY_COLUMNS}(Tuple([
        [field === :time ? record.time : getproperty(record.diagnostics, field) for record in records]
        for field in TRAJECTORY_COLUMNS
    ]))
end

"""Write explicit endpoint records as a headered numeric CSV file."""
function write_trajectory_csv(path::AbstractString, records)
    table = trajectory_table(records)
    open(path, "w") do io
        println(io, join(string.(TRAJECTORY_COLUMNS), ","))
        for row in eachindex(table.time)
            values = (getproperty(table, column)[row] for column in TRAJECTORY_COLUMNS)
            println(io, join(string.(values), ","))
        end
    end
    return path
end

"""Read and validate a CSV previously written by `write_trajectory_csv`."""
function read_trajectory_csv(path::AbstractString)
    lines = readlines(path)
    isempty(lines) && throw(ArgumentError("trajectory CSV must not be empty"))
    split(lines[1], ',') == collect(string.(TRAJECTORY_COLUMNS)) ||
        throw(ArgumentError("trajectory CSV columns do not match schema version $TRAJECTORY_SCHEMA_VERSION"))

    columns = [Float64[] for _ in TRAJECTORY_COLUMNS]
    for (offset, line) in enumerate(lines[2:end])
        line_number = offset + 1
        values = split(line, ',')
        length(values) == length(TRAJECTORY_COLUMNS) ||
            throw(ArgumentError("trajectory CSV line $line_number has the wrong column count"))
        for (column, value) in enumerate(values)
            try
                push!(columns[column], parse(Float64, value))
            catch
                throw(ArgumentError("trajectory CSV line $line_number has a nonnumeric value"))
            end
        end
    end
    isempty(columns[1]) && throw(ArgumentError("trajectory CSV must contain at least one record"))
    table = NamedTuple{TRAJECTORY_COLUMNS}(Tuple(columns))
    _validate_trajectory_table(table)
    return table
end

function _validate_trajectory_table(table)
    all(isfinite, table.time) || throw(ArgumentError("trajectory times must be finite"))
    all(diff(table.time) .> 0) || throw(ArgumentError("trajectory times must be strictly increasing"))
    for field in TRAJECTORY_COLUMNS[2:end]
        all(isfinite, getproperty(table, field)) ||
            throw(ArgumentError("trajectory column $field must be finite"))
    end
    all(value -> value >= 1 && isinteger(value), table.max_link_dimension) ||
        throw(ArgumentError("max_link_dimension values must be positive integers"))
    return nothing
end

"""Write caller-supplied provenance metadata as TOML with a schema version."""
function write_metadata_toml(path::AbstractString, metadata::AbstractDict)
    haskey(metadata, "schema_version") ||
        throw(ArgumentError("metadata must include schema_version"))
    metadata["schema_version"] == TRAJECTORY_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported metadata schema_version"))
    open(path, "w") do io
        TOML.print(io, metadata)
    end
    return path
end

"""Load metadata and require the current output schema version."""
function read_metadata_toml(path::AbstractString)
    metadata = TOML.parsefile(path)
    get(metadata, "schema_version", nothing) == TRAJECTORY_SCHEMA_VERSION ||
        throw(ArgumentError("unsupported or missing metadata schema_version"))
    return metadata
end
