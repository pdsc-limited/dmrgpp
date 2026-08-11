"""Construction of the two atomic product-state components.

The paper protocol evolves separate atomic `Up` and `Dn` impurity states and
averages scalar observables afterwards. These functions only translate the
already-validated data-only labels in `Model.jl` into an ITensorMPS product
state; they perform no evolution or configuration lookup.
"""

using ITensorMPS: MPS

"""Build one atomic SIAM product state from explicit sites and a model spec.

`impurity_state` is `:Up` or `:Dn`. The caller owns the site indices, making
QN policy and site ordering explicit. The state is a product MPS, not a
thermal sampling state and not an evolved trajectory.
"""
function initial_product_mps(model, sites, impurity_state::Symbol)
    length(sites) == length(model.site_order) ||
        throw(ArgumentError("site count must equal model bath_sites + 1"))
    labels = string.(initial_product_labels(model, impurity_state))
    return MPS(sites, labels)
end
