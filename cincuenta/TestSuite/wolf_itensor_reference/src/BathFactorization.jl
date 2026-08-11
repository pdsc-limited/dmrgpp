"""Diagnostics and truncated factors for finite-temperature bath kernels."""

using LinearAlgebra

"""Return Hermiticity and positive-semidefiniteness diagnostics for `matrix`.

The returned named tuple is data only: callers decide whether a failed check is
an error. Eigenvalues are evaluated only for square matrices.
"""
function hermitian_psd_diagnostics(
        matrix::AbstractMatrix;
        atol::Real = 1e-12,
        rtol::Real = 1e-10,
    )
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))

    square = size(matrix, 1) == size(matrix, 2)
    if !square
        return (
            is_square = false,
            is_hermitian = false,
            is_psd = false,
            hermiticity_error = Inf,
            minimum_eigenvalue = NaN,
            eigenvalues = Float64[],
            tolerance = NaN,
        )
    end

    scale = max(norm(matrix), 1.0)
    tolerance = float(atol) + float(rtol) * scale
    hermiticity_error = norm(matrix - adjoint(matrix))
    is_hermitian = hermiticity_error <= tolerance
    eigenvalues = eigvals(Hermitian(matrix))
    minimum_eigenvalue = isempty(eigenvalues) ? 0.0 : minimum(eigenvalues)
    is_psd = is_hermitian && minimum_eigenvalue >= -tolerance

    return (
        is_square = true,
        is_hermitian,
        is_psd,
        hermiticity_error,
        minimum_eigenvalue,
        eigenvalues,
        tolerance,
    )
end

"""Factor a Hermitian PSD matrix into a rank-capped factor.

Returns a named tuple containing `factor` such that `factor * factor'`
approximates `matrix`, plus retained/discarded spectral weights and a relative
reconstruction error. Eigenvector phases and ordering inside degenerate
subspaces are intentionally not part of the interface.
"""
function factorize_psd(
        matrix::AbstractMatrix;
        rank::Integer,
        atol::Real = 1e-12,
        rtol::Real = 1e-10,
    )
    diagnostics = hermitian_psd_diagnostics(matrix; atol, rtol)
    diagnostics.is_square || throw(ArgumentError("matrix must be square"))
    diagnostics.is_hermitian || throw(ArgumentError("matrix must be Hermitian"))
    diagnostics.is_psd || throw(ArgumentError("matrix must be positive semidefinite"))

    dimension = size(matrix, 1)
    0 <= rank <= dimension ||
        throw(ArgumentError("rank must lie between zero and the matrix dimension"))

    decomposition = eigen(Hermitian(matrix))
    order = sortperm(decomposition.values; rev = true)
    eigenvalues = decomposition.values[order]
    eigenvectors = decomposition.vectors[:, order]
    clamped_eigenvalues = max.(eigenvalues, zero(eltype(eigenvalues)))

    retained_eigenvalues = clamped_eigenvalues[1:rank]
    discarded_eigenvalues = clamped_eigenvalues[(rank + 1):end]
    factor = eigenvectors[:, 1:rank] * Diagonal(sqrt.(retained_eigenvalues))
    reconstruction = factor * adjoint(factor)
    total_weight = sum(clamped_eigenvalues)
    retained_weight = sum(retained_eigenvalues)
    discarded_weight = sum(discarded_eigenvalues)
    reconstruction_error = norm(matrix - reconstruction) / max(norm(matrix), 1.0)

    return (
        factor,
        rank,
        eigenvalues = clamped_eigenvalues,
        retained_eigenvalues,
        discarded_eigenvalues,
        total_weight,
        retained_weight,
        discarded_weight,
        reconstruction_error,
        diagnostics,
    )
end
