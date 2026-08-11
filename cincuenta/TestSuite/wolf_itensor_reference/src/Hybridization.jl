"""Numerical ingredients for the continuum hybridization in Wolf et al."""

"""Return the semicircular density of states on the support ``[-2, 2]``."""
function semicircular_dos(omega::Real)
    abs(omega) <= 2 || return 0.0
    return sqrt(max(0.0, 4.0 - float(omega)^2)) / (2 * pi)
end

"""Return ``1 / (exp(beta * (omega - mu)) + 1)`` without overflow."""
function fermi(omega::Real; beta::Real = 1.0, mu::Real = 0.0)
    beta > 0 || throw(ArgumentError("beta must be positive"))
    x = float(beta) * (float(omega) - float(mu))
    return x >= 0 ? exp(-x) / (1 + exp(-x)) : 1 / (1 + exp(x))
end

"""Return the Wolf cosine hopping ramp with final value one.

The physical protocol starts at ``t = 0``. Values at negative times are
clamped to zero so callers can safely construct time grids containing the
initial endpoint.
"""
function cosine_ramp(t::Real; t1::Real)
    t1 > 0 || throw(ArgumentError("t1 must be positive"))
    t <= 0 && return 0.0
    t >= t1 && return 1.0
    return (1 - cos(pi * float(t) / float(t1))) / 2
end

"""Integrate `integrand(omega) * A(omega)` over the semicircular band.

The substitution ``omega = 2 cos(theta)`` makes the endpoint behavior
smooth. Composite Simpson integration is deterministic, dependency-free, and
works for real or complex-valued integrands. `intervals` must be positive and
even.
"""
function integrate_semicircle(integrand::Function; intervals::Integer = 2048)
    intervals > 0 && iseven(intervals) ||
        throw(ArgumentError("intervals must be a positive even integer"))

    h = pi / intervals
    weighted(theta) = (2 / pi) * sin(theta)^2 * integrand(2 * cos(theta))
    total = weighted(0.0) + weighted(pi)
    for index in 1:(intervals - 1)
        total += (isodd(index) ? 4 : 2) * weighted(index * h)
    end
    return h * total / 3
end

"""Return the continuum lesser bath Green function at `delta_t = t - tprime`."""
function bath_green_lesser(
        delta_t::Real;
        beta::Real = 1.0,
        mu::Real = 0.0,
        intervals::Integer = 2048,
    )
    return im * integrate_semicircle(
        omega -> fermi(omega; beta, mu) * cis(-omega * delta_t);
        intervals,
    )
end

"""Return the continuum greater bath Green function at `delta_t = t - tprime`."""
function bath_green_greater(
        delta_t::Real;
        beta::Real = 1.0,
        mu::Real = 0.0,
        intervals::Integer = 2048,
    )
    return -im * integrate_semicircle(
        omega -> (1 - fermi(omega; beta, mu)) * cis(-omega * delta_t);
        intervals,
    )
end

"""Return the continuum lesser hybridization `Lambda<(t, tprime)`."""
function hybridization_lesser(
        t::Real,
        tprime::Real;
        t1::Real,
        beta::Real = 1.0,
        mu::Real = 0.0,
        intervals::Integer = 2048,
    )
    return cosine_ramp(t; t1) *
           bath_green_lesser(t - tprime; beta, mu, intervals) *
           cosine_ramp(tprime; t1)
end

"""Return the continuum greater hybridization `Lambda>(t, tprime)`."""
function hybridization_greater(
        t::Real,
        tprime::Real;
        t1::Real,
        beta::Real = 1.0,
        mu::Real = 0.0,
        intervals::Integer = 2048,
    )
    return cosine_ramp(t; t1) *
           bath_green_greater(t - tprime; beta, mu, intervals) *
           cosine_ramp(tprime; t1)
end

"""Build a continuum hybridization matrix for explicit time points.

`component` is either `:lesser` or `:greater`. No configuration or global
state is consulted; all model inputs are function arguments.
"""
function hybridization_matrix(
        times::AbstractVector{<:Real};
        component::Symbol,
        t1::Real,
        beta::Real = 1.0,
        mu::Real = 0.0,
        intervals::Integer = 2048,
    )
    kernel = if component === :lesser
        hybridization_lesser
    elseif component === :greater
        hybridization_greater
    else
        throw(ArgumentError("component must be :lesser or :greater"))
    end

    matrix = Matrix{ComplexF64}(undef, length(times), length(times))
    for column in eachindex(times), row in eachindex(times)
        matrix[row, column] = kernel(
            times[row], times[column];
            t1,
            beta,
            mu,
            intervals,
        )
    end
    return matrix
end
