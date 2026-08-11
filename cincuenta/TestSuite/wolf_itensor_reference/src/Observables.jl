"""Local scalar observables for explicit MPS states.

Observables take the state and physical MPS position explicitly. They do not
retain a model, time grid, or propagation history.
"""

using ITensors: inner
using ITensorMPS: expect, linkdims, maxlinkdim

"""Return `(nup, ndn, double_occupancy)` at one explicit MPS position."""
function impurity_occupations(state, impurity_position::Integer)
    1 <= impurity_position <= length(state) ||
        throw(ArgumentError("impurity_position must index the supplied state"))
    site = impurity_position:impurity_position
    return (
        nup = expect(state, "Nup"; sites = site)[1],
        ndn = expect(state, "Ndn"; sites = site)[1],
        double_occupancy = expect(state, "Nupdn"; sites = site)[1],
    )
end

"""Return the impurity double occupancy at an explicit MPS position."""
function impurity_double_occupancy(state, impurity_position::Integer)
    return impurity_occupations(state, impurity_position).double_occupancy
end

"""Return the equal-weight average of two scalar spin-component results."""
spin_average(up_component::Number, down_component::Number) =
    (up_component + down_component) / 2

"""Measure normalization, conserved charges, energy, and local impurity data.

All dependencies are explicit: `state` is the state to inspect,
`hamiltonian` is the Hamiltonian used for its energy measurement, and
`impurity_position` identifies the physical impurity site. The result is a
plain `NamedTuple`, suitable for writing or comparing outside this module.
`energy_imaginary_part` exposes numerical non-Hermiticity rather than hiding
it by silently taking a real part.
"""
function state_diagnostics(state, hamiltonian, impurity_position::Integer)
    norm_squared = real(inner(state, state))
    norm_squared > 0 || throw(ArgumentError("state must have nonzero norm"))

    impurity = impurity_occupations(state, impurity_position)
    nup = sum(expect(state, "Nup"))
    ndn = sum(expect(state, "Ndn"))
    energy = inner(state', hamiltonian, state; make_inds_match = false) / norm_squared

    return (
        norm = sqrt(norm_squared),
        total_particle_number = nup + ndn,
        spin_projection = (nup - ndn) / 2,
        energy = real(energy),
        energy_imaginary_part = imag(energy),
        impurity_nup = impurity.nup,
        impurity_ndn = impurity.ndn,
        impurity_double_occupancy = impurity.double_occupancy,
        max_link_dimension = maxlinkdim(state),
        link_dimensions = collect(linkdims(state)),
    )
end
