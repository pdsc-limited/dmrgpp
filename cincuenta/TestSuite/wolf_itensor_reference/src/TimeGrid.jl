"""Explicit time grids for midpoint propagators and bath-kernel factorization."""

"""Return factorization knots for midpoint propagation over `endpoints`.

The returned `points` interleave every endpoint with the midpoint of its
following interval. Factorizing a bath kernel on `points` lets a caller use
`factor[midpoint_indices, :]` directly as midpoint couplings. No factor-row
interpolation, refactorization, or time-grid choice is hidden here.

Only strictly increasing, finite endpoint times are accepted. The factorization
norm is presently unweighted because the planned production grid is uniform;
nonuniform quadrature weights must be introduced explicitly with a separate
weighted-factorization API rather than inferred by this helper.
"""
function midpoint_factorization_grid(endpoints::AbstractVector{<:Real})
    length(endpoints) >= 2 || throw(ArgumentError("at least two endpoint times are required"))
    all(isfinite, endpoints) || throw(ArgumentError("endpoint times must be finite"))
    all(diff(endpoints) .> 0) ||
        throw(ArgumentError("endpoint times must be strictly increasing"))

    endpoint_values = Float64.(endpoints)
    endpoint_count = length(endpoint_values)
    midpoint_values = (endpoint_values[1:(end - 1)] + endpoint_values[2:end]) ./ 2
    points = Vector{Float64}(undef, 2 * endpoint_count - 1)
    points[1:2:end] = endpoint_values
    points[2:2:end] = midpoint_values

    return (
        endpoints = endpoint_values,
        midpoints = midpoint_values,
        points,
        endpoint_indices = collect(1:2:length(points)),
        midpoint_indices = collect(2:2:length(points)),
    )
end

function _natural_spline_second_derivatives(
        times::AbstractVector{<:Real}, values::AbstractMatrix
    )
    count = length(times)
    size(values, 1) == count ||
        throw(DimensionMismatch("values must have one row per time"))
    second = zeros(ComplexF64, size(values))
    count <= 2 && return second

    spacing = diff(Float64.(times))
    interior_count = count - 2
    diagonal = ComplexF64[
        2 * (spacing[index] + spacing[index + 1]) for index in 1:interior_count
    ]
    off_diagonal = interior_count > 1 ?
        ComplexF64.(spacing[2:interior_count]) : ComplexF64[]
    right_hand_side = zeros(ComplexF64, interior_count, size(values, 2))
    for index in 1:interior_count
        knot = index + 1
        right_hand_side[index, :] .= 6 .* (
            (values[knot + 1, :] .- values[knot, :]) ./ spacing[knot] .-
            (values[knot, :] .- values[knot - 1, :]) ./ spacing[knot - 1]
        )
    end
    second[2:(end - 1), :] .=
        Tridiagonal(off_diagonal, diagonal, off_diagonal) \ right_hand_side
    return second
end

"""Interleave endpoint factors with causally spline-interpolated midpoints.

For interval `n`, a natural cubic spline is formed only from endpoint rows
available through `n+1`, and only that interval's midpoint is retained. This
matches stepwise time propagation: future endpoint rows can never modify a
midpoint coupling that has already been used.
"""
function causal_midpoint_factor_rows(
        endpoints::AbstractVector{<:Real}, endpoint_factor::AbstractMatrix
    )
    length(endpoints) >= 2 || throw(ArgumentError("at least two endpoints are required"))
    size(endpoint_factor, 1) == length(endpoints) ||
        throw(DimensionMismatch("endpoint_factor must have one row per endpoint"))
    all(isfinite, endpoints) || throw(ArgumentError("endpoints must be finite"))
    all(diff(endpoints) .> 0) ||
        throw(ArgumentError("endpoints must be strictly increasing"))

    endpoint_values = ComplexF64.(endpoint_factor)
    rows = zeros(ComplexF64, 2 * length(endpoints) - 1, size(endpoint_values, 2))
    rows[1:2:end, :] .= endpoint_values
    for interval in 1:(length(endpoints) - 1)
        prefix = 1:(interval + 1)
        second = _natural_spline_second_derivatives(
            endpoints[prefix], endpoint_values[prefix, :]
        )
        spacing = endpoints[interval + 1] - endpoints[interval]
        rows[2 * interval, :] .=
            (endpoint_values[interval, :] .+ endpoint_values[interval + 1, :]) ./ 2 .-
            spacing^2 .* (second[end - 1, :] .+ second[end, :]) ./ 16
    end
    return rows
end

"""Return a copy of selected factor rows after checking their grid indices."""
function factor_rows(factor::AbstractMatrix, indices::AbstractVector{<:Integer})
    all(index -> checkbounds(Bool, axes(factor, 1), index), indices) ||
        throw(BoundsError(factor, (indices, :)))
    return copy(factor[indices, :])
end
