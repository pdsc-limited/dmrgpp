"""Data-only specifications for the factorized star SIAM.

This file deliberately contains no ITensor objects. It turns the two
factorization matrices into explicit bath-site data that the later MPO layer
can consume.
"""

"""Return the MPS position of the central impurity for an even bath size."""
function central_impurity_position(bath_sites::Integer)
    bath_sites > 0 || throw(ArgumentError("bath_sites must be positive"))
    iseven(bath_sites) || throw(ArgumentError("bath_sites must be even"))
    return bath_sites ÷ 2 + 1
end

"""Return the physical-site order with bath labels and a central impurity.

Bath orbitals retain their factor-column order. The returned vector uses `0`
for the impurity and positive integers for bath orbitals, so it is independent
of any MPS or site-type representation.
"""
function central_star_order(bath_sites::Integer)
    impurity_position = central_impurity_position(bath_sites)
    order = collect(1:bath_sites)
    insert!(order, impurity_position, 0)
    return order
end

"""Combine lesser and greater kernel factors into a factorized bath spec.

Rows label explicit time-grid points and columns label bath orbitals. Lesser
columns describe initially occupied bath sites; greater columns describe
initially empty bath sites. The returned named tuple is immutable data only.
"""
function factorized_bath_spec(
        lesser_factor::AbstractMatrix,
        greater_factor::AbstractMatrix;
        final_bath_potential::Real = 0.0,
    )
    size(lesser_factor, 1) == size(greater_factor, 1) ||
        throw(ArgumentError("lesser and greater factors must share a time-grid dimension"))
    lesser_rank = size(lesser_factor, 2)
    greater_rank = size(greater_factor, 2)
    lesser_rank == greater_rank ||
        throw(ArgumentError("lesser and greater factors must have equal rank"))
    lesser_rank > 0 || throw(ArgumentError("factor ranks must be positive"))

    bath_sites = lesser_rank + greater_rank
    return (
        time_points = size(lesser_factor, 1),
        bath_sites,
        lesser_rank,
        greater_rank,
        couplings = hcat(complex.(lesser_factor), complex.(greater_factor)),
        bath_potentials = fill(float(final_bath_potential), bath_sites),
        initial_occupations = vcat(fill(:UpDn, lesser_rank), fill(:Emp, greater_rank)),
    )
end

"""Return an immutable SIAM specification from explicit bath data.

`interaction` is the coefficient of
`(Nup - 1/2) * (Ndn - 1/2)`. All numerical dependencies are arguments;
configuration parsing and MPS construction intentionally belong elsewhere.
"""
function siam_model_spec(
        interaction::Real,
        bath;
        chemical_potential::Real = 0.0,
    )
    isfinite(interaction) || throw(ArgumentError("interaction must be finite"))
    isfinite(chemical_potential) || throw(ArgumentError("chemical_potential must be finite"))
    bath.bath_sites == length(bath.bath_potentials) ||
        throw(ArgumentError("bath potentials must have one entry per bath site"))
    bath.bath_sites == length(bath.initial_occupations) ||
        throw(ArgumentError("bath occupations must have one entry per bath site"))
    size(bath.couplings) == (bath.time_points, bath.bath_sites) ||
        throw(ArgumentError("couplings must have shape (time_points, bath_sites)"))

    return (
        interaction = float(interaction),
        chemical_potential = float(chemical_potential),
        bath,
        site_order = central_star_order(bath.bath_sites),
        impurity_position = central_impurity_position(bath.bath_sites),
    )
end

"""Return couplings for one one-based factorization-grid index."""
function couplings_at(model, time_index::Integer)
    checkbounds(Bool, axes(model.bath.couplings, 1), time_index) ||
        throw(BoundsError(model.bath.couplings, (time_index, :)))
    return copy(model.bath.couplings[time_index, :])
end

"""Return a product-state label vector for one atomic impurity component.

Labels use the conventional electron-state names `:Emp`, `:Up`, `:Dn`, and
`:UpDn`, but are not ITensor objects. The bath part comes solely from `model`.
"""
function initial_product_labels(model, impurity_state::Symbol)
    impurity_state in (:Up, :Dn) ||
        throw(ArgumentError("impurity_state must be :Up or :Dn"))
    labels = copy(model.bath.initial_occupations)
    insert!(labels, model.impurity_position, impurity_state)
    return labels
end
