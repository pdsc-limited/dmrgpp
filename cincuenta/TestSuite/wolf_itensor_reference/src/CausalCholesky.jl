"""Causal fixed-rank Cholesky factors for nonequilibrium bath kernels.

This implements the optimized low-rank construction of Gramsch et al.,
Phys. Rev. B 88, 235106 (2013), Eqs. 56--63. The target convention is
`kernel[n,j] = -im * Lambda_lesser(t_n,t_j)` (or analogously
`im * Lambda_greater`), so that `kernel ≈ factor * factor'`.
"""

using LinearAlgebra

function _causal_optimal_update(
        design::AbstractMatrix,
        target::AbstractVector,
        diagonal_target::Real,
    )
    rank = size(design, 2)
    size(design, 1) == length(target) ||
        throw(DimensionMismatch("design rows and target length must agree"))
    rank > 0 || return ComplexF64[]

    d = max(float(diagonal_target), 0.0)
    gram = Hermitian(adjoint(design) * design)
    rhs = adjoint(design) * target
    decomposition = eigen(gram)
    eigenvalues = max.(Float64.(decomposition.values), 0.0)
    eigenvectors = decomposition.vectors
    transformed_rhs = adjoint(eigenvectors) * rhs

    function coefficients_at(mu)
        denominators = eigenvalues .+ (mu - d)
        return transformed_rhs ./ denominators
    end
    function norm_squared_at(mu)
        denominators = eigenvalues .+ (mu - d)
        return sum(abs2.(transformed_rhs) ./ denominators .^ 2)
    end

    # The near-null eigenspace is a free direction for the off-diagonal fit.
    # Use it to satisfy the diagonal constraint directly (the trust-region
    # "hard case") rather than bisecting arbitrarily close to a pole.
    largest_eigenvalue = eigenvalues[end]
    hard = largest_eigenvalue > 0 ?
        eigenvalues .< 1e-6 * largest_eigenvalue : trues(rank)
    if any(hard) && sum(abs2, transformed_rhs[hard]) < 1e-6 * max(d, 1.0)
        soft = .!hard
        coefficients = zeros(ComplexF64, rank)
        coefficients[soft] .= transformed_rhs[soft] ./ eigenvalues[soft]
        remainder = d - sum(abs2, coefficients[soft])
        if remainder > 0
            hard_indices = findall(hard)
            amplitude = sqrt(remainder / length(hard_indices))
            for index in hard_indices
                value = transformed_rhs[index]
                phase = abs(value) > 0 ? value / abs(value) : one(value)
                coefficients[index] = amplitude * phase
            end
            candidate = eigenvectors * coefficients
            candidate_norm = real(dot(candidate, candidate))
            if all(isfinite, candidate) &&
                    abs(candidate_norm - d) < 1e-6 * max(d, 1.0)
                return candidate
            end
        end
    end

    linear_coefficients = coefficients_at(d)
    linear_candidate = eigenvectors * linear_coefficients
    linear_norm = real(dot(linear_candidate, linear_candidate))
    all(isfinite, linear_candidate) || return zeros(ComplexF64, rank)

    norm_at_d = norm_squared_at(d)
    isfinite(norm_at_d) || return linear_candidate

    if norm_at_d >= d
        lower = d
        upper = d
        step = 0.5 * max(d, 1.0)
        bracketed = false
        for _ in 1:60
            upper += step
            value = norm_squared_at(upper)
            if isfinite(value) && value - upper <= 0
                bracketed = true
                break
            end
            step *= 1.5
        end
        bracketed || return linear_candidate
    else
        upper = d
        current = d
        step = 0.5 * max(d, 1.0)
        lower = 0.0
        bracketed = false
        for _ in 1:60
            current = max(current - step, 0.0)
            value = norm_squared_at(current)
            if isfinite(value) && (value - current >= 0 || current == 0.0)
                lower = current
                bracketed = true
                break
            end
            step *= 1.5
        end
        bracketed || return linear_candidate
    end

    for _ in 1:100
        midpoint = (lower + upper) / 2
        value = norm_squared_at(midpoint)
        if isfinite(value) && value - midpoint >= 0
            lower = midpoint
        else
            upper = midpoint
        end
        upper - lower < 1e-14 * max(upper, 1.0) && break
    end

    final_candidate = eigenvectors * coefficients_at(lower)
    final_norm = real(dot(final_candidate, final_candidate))
    bound = 10 * max(d, linear_norm, norm_at_d, 1.0)
    if !all(isfinite, final_candidate) || final_norm > bound ||
            abs(final_norm - lower) > 0.05 * max(lower, 1.0)
        return linear_candidate
    end
    return final_candidate
end

"""Return the causal optimized rank-`rank` factor of a Hermitian bath kernel.

The first time is the atomic-limit point and must have a zero row and column.
Rows after the first `rank` positive-time rows use all previously constructed
rows in the joint off-diagonal/diagonal optimization. Consequently, extending
the target matrix cannot alter an already constructed factor row.
"""
function factorize_causal_cholesky(
        kernel::AbstractMatrix;
        rank::Integer,
        floor_rtol::Real = 1e-10,
        atol::Real = 1e-11,
        rtol::Real = 1e-10,
    )
    size(kernel, 1) == size(kernel, 2) ||
        throw(ArgumentError("kernel must be square"))
    dimension = size(kernel, 1)
    rank >= 0 || throw(ArgumentError("rank must be nonnegative"))
    floor_rtol >= 0 || throw(ArgumentError("floor_rtol must be nonnegative"))
    atol >= 0 || throw(ArgumentError("atol must be nonnegative"))
    rtol >= 0 || throw(ArgumentError("rtol must be nonnegative"))

    complex_kernel = ComplexF64.(kernel)
    scale = max(norm(complex_kernel), 1.0)
    tolerance = float(atol) + float(rtol) * scale
    norm(complex_kernel - adjoint(complex_kernel)) <= tolerance ||
        throw(ArgumentError("kernel must be Hermitian"))
    if dimension > 0
        max(norm(complex_kernel[1, :]), norm(complex_kernel[:, 1])) <= tolerance ||
            throw(ArgumentError("the atomic-limit row and column must vanish"))
    end

    factor = zeros(ComplexF64, dimension, rank)
    maximum_diagonal_seen = 0.0
    for physical_step in 1:(dimension - 1)
        row = physical_step + 1
        diagonal = real(complex_kernel[row, row])
        maximum_diagonal_seen = max(maximum_diagonal_seen, abs(diagonal))

        if physical_step <= rank
            column = physical_step
            for previous_column in 1:(column - 1)
                pivot_row = previous_column + 1
                value = complex_kernel[row, pivot_row]
                for k in 1:(previous_column - 1)
                    value -= factor[row, k] * conj(factor[pivot_row, k])
                end
                pivot = factor[pivot_row, previous_column]
                factor[row, previous_column] = abs(pivot) < 1e-14 ?
                    zero(ComplexF64) : value / conj(pivot)
            end
            residual = diagonal - sum(abs2, factor[row, 1:(column - 1)])
            floor = floor_rtol * max(maximum_diagonal_seen, 1e-300)
            factor[row, column] = sqrt(max(residual, floor))
        elseif rank > 0
            # For factor*factor', the standard least-squares design matrix is
            # conjugate(previous factor rows). This conjugation is essential
            # for kernels carrying a genuine complex phase.
            target = complex_kernel[row, 2:physical_step]
            design = conj.(factor[2:physical_step, :])
            factor[row, :] .= _causal_optimal_update(design, target, diagonal)
        end
    end

    reconstruction = factor * adjoint(factor)
    denominator = max(norm(complex_kernel), 1.0)
    return (
        factor,
        rank,
        reconstruction_error = norm(complex_kernel - reconstruction) / denominator,
        maximum_absolute_error = maximum(abs, complex_kernel - reconstruction; init = 0.0),
        floor_rtol = float(floor_rtol),
    )
end
