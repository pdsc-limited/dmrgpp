"""ITensorMPS construction for a static time slice of the factorized star SIAM.

The data-only model specification lives in `Model.jl`. Every function here
receives that model (and, where relevant, a time-grid index) explicitly; this
module neither reads configuration files nor evolves an MPS.
"""

using ITensors: OpSum
using ITensorMPS: MPO, siteinds

"""Return QN-conserving spinful-electron site indices for `model`.

The index order follows `model.site_order`, whose central position is the
impurity. The physical bath labels are retained only in `model.site_order`;
ITensor site numbers are one-based MPS positions.
"""
function electron_sites(model; conserve_qns::Bool = true)
    return siteinds("Electron", length(model.site_order); conserve_qns)
end

"""Return the MPS position of a physical bath orbital label in `model`."""
function bath_position(model, bath_label::Integer)
    1 <= bath_label <= model.bath.bath_sites ||
        throw(ArgumentError("bath_label must be in 1:$(model.bath.bath_sites)"))
    return findfirst(==(bath_label), model.site_order)::Int
end

"""Return an `OpSum` for one factorization-grid time slice of `model`.

The shifted interaction is expanded as
`U*Nup*Ndn + (-U/2-mu)*(Nup+Ndn) + U/4*Id` at the impurity. For a bath
coupling `V`, each spin has `V*Cdag_imp*C_bath + conj(V)*Cdag_bath*C_imp`.
The explicit conjugate terms make the operator Hermitian for complex factors.
"""
function siam_opsum(model, time_index::Integer)
    couplings = couplings_at(model, time_index)
    impurity = model.impurity_position
    interaction = model.interaction
    chemical_potential = model.chemical_potential

    terms = OpSum()
    terms += interaction, "Nup", impurity, "Ndn", impurity
    terms += -interaction / 2 - chemical_potential, "Nup", impurity
    terms += -interaction / 2 - chemical_potential, "Ndn", impurity
    terms += interaction / 4, "Id", impurity

    for bath_label in 1:model.bath.bath_sites
        bath = bath_position(model, bath_label)
        potential = model.bath.bath_potentials[bath_label]
        potential == 0 || begin
            terms += potential, "Nup", bath
            terms += potential, "Ndn", bath
        end

        coupling = couplings[bath_label]
        for (creation, annihilation) in (("Cdagup", "Cup"), ("Cdagdn", "Cdn"))
            terms += coupling, creation, impurity, annihilation, bath
            terms += conj(coupling), creation, bath, annihilation, impurity
        end
    end
    return terms
end

"""Build the static SIAM MPO for one explicit factorization-grid index.

`sites` is an explicit dependency so callers can choose QN policy and reuse
an already-created site-index array. Its length must match the model order.
"""
function siam_mpo(model, sites, time_index::Integer)
    length(sites) == length(model.site_order) ||
        throw(ArgumentError("site count must equal model bath_sites + 1"))
    return MPO(siam_opsum(model, time_index), sites)
end
