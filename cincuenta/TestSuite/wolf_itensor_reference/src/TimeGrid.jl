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

"""Return a copy of selected factor rows after checking their grid indices."""
function factor_rows(factor::AbstractMatrix, indices::AbstractVector{<:Integer})
    all(index -> checkbounds(Bool, axes(factor, 1), index), indices) ||
        throw(BoundsError(factor, (indices, :)))
    return copy(factor[indices, :])
end
