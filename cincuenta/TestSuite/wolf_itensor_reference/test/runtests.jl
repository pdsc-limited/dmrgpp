using Test
using ITensors
using ITensorMPS

include(joinpath(@__DIR__, "..", "src", "WolfITensorReference.jl"))
using .WolfITensorReference

const ROOT = normpath(joinpath(@__DIR__, ".."))

@testset "Wolf benchmark configuration contract" begin
    for (filename, expected_u) in (("wolf_u4.toml", 4.0), ("wolf_u10.toml", 10.0))
        config = load_benchmark_config(joinpath(ROOT, "configs", filename))
        @test isempty(validate_benchmark_config(config))
        @test config["model"]["beta"] == 1.0
        @test config["model"]["U"] == expected_u
        @test config["time_grid"]["status"] == "provisional"
        @test config["time_grid"]["ramp_duration"] == 0.25
        @test contains(config["time_grid"]["ramp_duration_status"], "provisional")
        @test config["initial_state"]["status"] == "resolved"
        @test contains(config["initial_state"]["impurity_protocol"], "pure atomic")
        @test contains(config["initial_state"]["bath_protocol"], "doubly occupied")
        @test !benchmark_is_runnable(config)

        for entry in config["bath"]["discretizations"]
            @test entry["Lb"] >= 10
            @test iseven(entry["Lb"])
            @test entry["lesser_rank"] == entry["Lb"] ÷ 2
            @test entry["greater_rank"] == entry["Lb"] ÷ 2
        end
    end
end

@testset "Continuum hybridization ingredients" begin
    @test semicircular_dos(-2.0) == 0.0
    @test semicircular_dos(2.0) == 0.0
    @test semicircular_dos(3.0) == 0.0
    @test semicircular_dos(0.0) ≈ 1 / pi

    @test integrate_semicircle(_ -> 1.0; intervals = 256) ≈ 1.0 atol = 1e-10
    @test integrate_semicircle(identity; intervals = 256) ≈ 0.0 atol = 1e-12
    @test integrate_semicircle(omega -> omega^2; intervals = 256) ≈ 1.0 atol = 1e-9
    @test_throws ArgumentError integrate_semicircle(identity; intervals = 255)

    for omega in (-3.0, -0.7, 0.0, 0.7, 3.0)
        @test fermi(-omega) ≈ 1 - fermi(omega) atol = 1e-15
    end
    @test_throws ArgumentError fermi(0.0; beta = 0.0)

    @test cosine_ramp(-0.1; t1 = 0.25) == 0.0
    @test cosine_ramp(0.0; t1 = 0.25) == 0.0
    @test cosine_ramp(0.125; t1 = 0.25) ≈ 0.5 atol = 1e-15
    @test cosine_ramp(0.25; t1 = 0.25) == 1.0
    @test cosine_ramp(2.0; t1 = 0.25) == 1.0
    @test_throws ArgumentError cosine_ramp(0.0; t1 = 0.0)
end

@testset "Continuum finite-temperature Green functions" begin
    intervals = 512
    lesser_zero = bath_green_lesser(0.0; intervals)
    greater_zero = bath_green_greater(0.0; intervals)
    @test lesser_zero ≈ 0.5im atol = 1e-12
    @test greater_zero ≈ -0.5im atol = 1e-12
    @test greater_zero - lesser_zero ≈ -im atol = 1e-12

    for delta_t in (0.1, 1.7, 5.0)
        lesser = bath_green_lesser(delta_t; intervals)
        greater = bath_green_greater(delta_t; intervals)
        @test lesser ≈ -conj(bath_green_lesser(-delta_t; intervals)) atol = 1e-12
        @test greater ≈ -conj(bath_green_greater(-delta_t; intervals)) atol = 1e-12
    end
end

@testset "Hybridization matrices" begin
    times = [0.0, 0.125, 0.25, 1.0]
    lesser = hybridization_matrix(times; component = :lesser, t1 = 0.25, intervals = 512)
    greater = hybridization_matrix(times; component = :greater, t1 = 0.25, intervals = 512)

    @test size(lesser) == (length(times), length(times))
    @test lesser[1, :] == zeros(ComplexF64, length(times))
    @test lesser[:, 1] == zeros(ComplexF64, length(times))
    @test lesser ≈ -adjoint(lesser) atol = 1e-12
    @test greater ≈ -adjoint(greater) atol = 1e-12
    @test lesser[3, 3] ≈ 0.5im atol = 1e-12
    @test greater[3, 3] ≈ -0.5im atol = 1e-12
    @test_throws ArgumentError hybridization_matrix(times; component = :invalid, t1 = 0.25)
end

@testset "Positive-semidefinite bath factorization" begin
    synthetic = ComplexF64[2 1im 0; -1im 2 0; 0 0 0.5]
    diagnostics = hermitian_psd_diagnostics(synthetic)
    @test diagnostics.is_square
    @test diagnostics.is_hermitian
    @test diagnostics.is_psd
    @test diagnostics.minimum_eigenvalue ≥ 0

    full = factorize_psd(synthetic; rank = 3)
    @test full.factor * full.factor' ≈ synthetic atol = 1e-12
    @test full.discarded_weight ≈ 0.0 atol = 1e-12
    @test full.reconstruction_error ≤ 1e-12

    rank_one = factorize_psd(synthetic; rank = 1)
    rank_two = factorize_psd(synthetic; rank = 2)
    @test rank_one.discarded_weight ≥ rank_two.discarded_weight
    @test rank_one.reconstruction_error ≥ rank_two.reconstruction_error
    @test size(rank_one.factor) == (3, 1)

    zero_rank = factorize_psd(synthetic; rank = 0)
    @test size(zero_rank.factor) == (3, 0)
    @test zero_rank.retained_weight == 0

    nonsquare = hermitian_psd_diagnostics(ones(2, 3))
    @test !nonsquare.is_square
    @test !nonsquare.is_psd
    @test_throws ArgumentError factorize_psd(ones(2, 3); rank = 1)
    @test_throws ArgumentError factorize_psd(ComplexF64[1 1; 0 1]; rank = 1)
    @test_throws ArgumentError factorize_psd(ComplexF64[1 0; 0 -0.1]; rank = 1)
    @test_throws ArgumentError factorize_psd(synthetic; rank = 4)
end

@testset "Finite-temperature kernel factorization" begin
    times = [0.0, 0.125, 0.25, 0.5, 1.0]
    lesser = hybridization_matrix(times; component = :lesser, t1 = 0.25, intervals = 512)
    greater = hybridization_matrix(times; component = :greater, t1 = 0.25, intervals = 512)

    for kernel in (-im * lesser, im * greater)
        diagnostics = hermitian_psd_diagnostics(kernel; atol = 1e-11)
        @test diagnostics.is_hermitian
        @test diagnostics.is_psd

        full = factorize_psd(kernel; rank = length(times), atol = 1e-11)
        truncated = factorize_psd(kernel; rank = 2, atol = 1e-11)
        @test full.factor * full.factor' ≈ kernel atol = 1e-11
        @test full.discarded_weight ≤ 1e-11
        @test truncated.discarded_weight ≥ full.discarded_weight
        @test truncated.reconstruction_error ≥ full.reconstruction_error
    end
end

@testset "Factorized star model specification" begin
    lesser_factor = ComplexF64[1 0.2im; 0.5 0.3; 0.1im -0.2]
    greater_factor = ComplexF64[0.4 -0.1im; -0.2 0.6; 0.3im 0.1]
    bath = factorized_bath_spec(lesser_factor, greater_factor)

    @test bath.time_points == 3
    @test bath.bath_sites == 4
    @test bath.lesser_rank == 2
    @test bath.greater_rank == 2
    @test bath.couplings == hcat(lesser_factor, greater_factor)
    @test bath.bath_potentials == zeros(4)
    @test bath.initial_occupations == [:UpDn, :UpDn, :Emp, :Emp]
    @test central_impurity_position(4) == 3
    @test central_star_order(4) == [1, 2, 0, 3, 4]
    @test_throws ArgumentError central_impurity_position(3)
    @test_throws ArgumentError factorized_bath_spec(lesser_factor, greater_factor[:, 1:1])
    @test_throws ArgumentError factorized_bath_spec(lesser_factor, greater_factor[1:2, :])

    model = siam_model_spec(4.0, bath; chemical_potential = 0.0)
    @test model.interaction == 4.0
    @test model.chemical_potential == 0.0
    @test model.impurity_position == 3
    @test model.site_order == [1, 2, 0, 3, 4]
    @test couplings_at(model, 2) == bath.couplings[2, :]
    @test couplings_at(model, 2) !== bath.couplings[2, :]
    @test initial_product_labels(model, :Up) == [:UpDn, :UpDn, :Up, :Emp, :Emp]
    @test initial_product_labels(model, :Dn) == [:UpDn, :UpDn, :Dn, :Emp, :Emp]
    @test_throws BoundsError couplings_at(model, 4)
    @test_throws ArgumentError initial_product_labels(model, :Emp)

    times = [0.0, 0.125, 0.25, 0.5, 1.0]
    lesser = -im * hybridization_matrix(times; component = :lesser, t1 = 0.25, intervals = 512)
    greater = im * hybridization_matrix(times; component = :greater, t1 = 0.25, intervals = 512)
    paper_bath = factorized_bath_spec(
        factorize_psd(lesser; rank = 5, atol = 1e-11).factor,
        factorize_psd(greater; rank = 5, atol = 1e-11).factor,
    )
    @test paper_bath.bath_sites == 10
    @test size(paper_bath.couplings) == (length(times), 10)
end

@testset "Static factorized SIAM MPO" begin
    # This is a static Lb=10 construction check, not an MPS trajectory.
    lesser_factor = fill(0.2 + 0.1im, 2, 5)
    greater_factor = fill(-0.3 + 0.05im, 2, 5)
    bath = factorized_bath_spec(lesser_factor, greater_factor)
    model = siam_model_spec(4.0, bath; chemical_potential = 0.0)
    sites = electron_sites(model)

    @test length(sites) == 11
    @test model.site_order == [1, 2, 3, 4, 5, 0, 6, 7, 8, 9, 10]
    @test bath_position(model, 1) == 1
    @test bath_position(model, 5) == 5
    @test bath_position(model, 6) == 7
    @test bath_position(model, 10) == 11
    @test_throws ArgumentError bath_position(model, 0)
    @test_throws ArgumentError bath_position(model, 11)

    terms = siam_opsum(model, 2)
    hamiltonian = siam_mpo(model, sites, 2)
    @test terms isa OpSum
    @test length(hamiltonian) == length(sites)
    @test all(j -> hasind(hamiltonian[j], sites[j]), eachindex(sites))
    @test_throws BoundsError siam_opsum(model, 3)
    @test_throws ArgumentError siam_mpo(model, sites[1:10], 2)

    labels = string.(initial_product_labels(model, :Up))
    product_state = MPS(sites, labels)
    # All hopping terms have zero expectation in this occupation product state.
    # The shifted interaction on an up impurity is -U/4 = -1.
    @test inner(product_state', hamiltonian, product_state; make_inds_match = false) ≈ -1.0 atol = 1e-12
end

@testset "Explicit MPS Krylov algebra" begin
    # Static Lb=10 algebra checks only; this is not a propagation trajectory.
    bath = factorized_bath_spec(fill(0.2 + 0.1im, 1, 5), fill(-0.1 + 0.05im, 1, 5))
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    state = initial_product_mps(model, sites, :Up)
    hamiltonian = siam_mpo(model, sites, 1)

    @test mps_norm(state) ≈ 1.0 atol = 1e-12
    normalized = normalized_mps(2.0 * state)
    @test mps_norm(normalized) ≈ 1.0 atol = 1e-12
    @test abs(mps_overlap(state, normalized)) ≈ 1.0 atol = 1e-12

    action = mps_action(hamiltonian, state; cutoff = 0.0, maxdim = 100)
    @test mps_overlap(state, action) ≈ -1.0 atol = 1e-12
    @test mps_norm(action) > 1.0

    combination = mps_linear_combination(
        MPS[state, action],
        ComplexF64[1, im];
        cutoff = 0.0,
        maxdim = 100,
    )
    @test mps_overlap(state, combination) ≈ 1.0 - im atol = 1e-12

    projection = mps_overlap(state, action)
    residual = mps_subtract_projection(
        action,
        MPS[state],
        ComplexF64[projection];
        cutoff = 0.0,
        maxdim = 100,
    )
    @test abs(mps_overlap(state, residual)) ≤ 1e-10
    @test mps_norm(residual) > 0.0
    @test mps_basis_overlaps(MPS[state], action) == ComplexF64[projection]

    @test_throws ArgumentError mps_action(hamiltonian, state; cutoff = -1e-12, maxdim = 100)
    @test_throws ArgumentError mps_linear_combination(
        MPS[state], ComplexF64[]; cutoff = 0.0, maxdim = 100
    )
    @test_throws ArgumentError mps_subtract_projection(
        action, MPS[state], ComplexF64[]; cutoff = 0.0, maxdim = 100
    )
end

@testset "Global MPS Arnoldi basis" begin
    # Static Lb=10 basis construction only; no exponential or trajectory.
    bath = factorized_bath_spec(fill(0.2 + 0.1im, 1, 5), fill(-0.1 + 0.05im, 1, 5))
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    state = initial_product_mps(model, sites, :Up)
    hamiltonian = siam_mpo(model, sites, 1)
    arnoldi = global_mps_arnoldi_basis(
        state,
        hamiltonian;
        krylovdim = 3,
        action_cutoff = 0.0,
        action_maxdim = 100,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 100,
        breakdown_tolerance = 1e-12,
    )

    m = arnoldi.accepted_dimension
    @test m == 3
    @test !arnoldi.breakdown
    @test length(arnoldi.basis) == m
    @test arnoldi.next_vector !== nothing
    @test size(arnoldi.hessenberg) == (m + 1, m)
    @test arnoldi.input_norm ≈ 1.0 atol = 1e-12
    @test all(isapprox(mps_norm(vector), 1.0; atol = 1e-10) for vector in arnoldi.basis)
    overlaps = [mps_overlap(left, right) for left in arnoldi.basis, right in arnoldi.basis]
    @test overlaps ≈ Matrix{ComplexF64}(LinearAlgebra.I, m, m) atol = 1e-9
    @test maximum(abs, arnoldi.second_pass_defects) ≤ 1e-9

    # The first Arnoldi relation uses only q1 and q2, both stored in `basis`.
    reconstructed_action = mps_linear_combination(
        arnoldi.basis[1:2],
        arnoldi.hessenberg[1:2, 1];
        cutoff = 0.0,
        maxdim = 100,
    )
    direct_action = mps_action(hamiltonian, arnoldi.basis[1]; cutoff = 0.0, maxdim = 100)
    relation_residual = mps_subtract_projection(
        direct_action,
        MPS[reconstructed_action],
        ComplexF64[1];
        cutoff = 0.0,
        maxdim = 100,
    )
    @test mps_norm(relation_residual) ≤ 1e-9

    @test_throws ArgumentError global_mps_arnoldi_basis(
        state, hamiltonian;
        krylovdim = 0,
        action_cutoff = 0.0,
        action_maxdim = 100,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 100,
        breakdown_tolerance = 1e-12,
    )
    @test_throws ArgumentError global_mps_arnoldi_basis(
        state, hamiltonian;
        krylovdim = 2,
        action_cutoff = -1e-12,
        action_maxdim = 100,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 100,
        breakdown_tolerance = 1e-12,
    )
end

@testset "One midpoint global MPS Krylov step" begin
    # This is an Lb=10 static-Hamiltonian primitive test, not a trajectory or
    # a Wolf benchmark result.
    bath = factorized_bath_spec(fill(0.2 + 0.1im, 1, 5), fill(-0.1 + 0.05im, 1, 5))
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    state = initial_product_mps(model, sites, :Up)
    hamiltonian = siam_mpo(model, sites, 1)
    controls = (
        krylovdim = 3,
        action_cutoff = 0.0,
        action_maxdim = 100,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 100,
        combination_cutoff = 0.0,
        combination_maxdim = 100,
        breakdown_tolerance = 1e-12,
    )

    unchanged = midpoint_global_krylov_step(state, hamiltonian, 0.0; controls...)
    @test unchanged.state !== state
    @test unchanged.state ≈ state atol = 1e-14
    @test unchanged.diagnostics.accepted_dimension == 0
    @test unchanged.diagnostics.projected_residual == 0.0
    @test unchanged.diagnostics.norm_drift == 0.0

    evolved = midpoint_global_krylov_step(state, hamiltonian, 0.02; controls...)
    diagnostics = evolved.diagnostics
    @test evolved.state !== state
    @test diagnostics.accepted_dimension == 3
    @test !diagnostics.breakdown
    @test size(diagnostics.hessenberg) == (4, 3)
    @test diagnostics.projected_hermiticity_defect ≤ 1e-9
    @test maximum(abs, diagnostics.second_pass_defects) ≤ 1e-9
    @test diagnostics.projected_residual ≥ 0.0
    @test diagnostics.input_norm ≈ 1.0 atol = 1e-12
    @test diagnostics.output_norm ≈ 1.0 atol = 1e-9
    @test abs(diagnostics.norm_drift) ≤ 1e-9
    @test diagnostics.controls == controls

    @test_throws ArgumentError midpoint_global_krylov_step(state, hamiltonian, Inf; controls...)
    invalid_controls = merge(controls, (combination_maxdim = 0,))
    @test_throws ArgumentError midpoint_global_krylov_step(
        state, hamiltonian, 0.02; invalid_controls...
    )
end

@testset "Global Krylov remote first-order amplitude" begin
    # Lb=10 U=0 plumbing gate only. A down electron transfers from the remote
    # left bath site at MPS position 1 to an initially-Up impurity at position
    # 6. An occupation would change only at O(dt^2), so test this Fock-state
    # amplitude directly at O(dt).
    bath = factorized_bath_spec(fill(0.2 + 0.1im, 1, 5), fill(-0.1 + 0.05im, 1, 5))
    model = siam_model_spec(0.0, bath)
    sites = electron_sites(model)
    initial = initial_product_mps(model, sites, :Up)
    target_labels = initial_product_labels(model, :Up)
    source = bath_position(model, 1)
    target_labels[source] = :Up
    target_labels[model.impurity_position] = :UpDn
    target = MPS(sites, string.(target_labels))
    hamiltonian = siam_mpo(model, sites, 1)
    direct_matrix_element = mps_overlap(
        target,
        mps_action(hamiltonian, initial; cutoff = 0.0, maxdim = 1000),
    )
    controls = (
        krylovdim = 5,
        action_cutoff = 0.0,
        action_maxdim = 1000,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 1000,
        combination_cutoff = 0.0,
        combination_maxdim = 1000,
        breakdown_tolerance = 1e-12,
    )
    large_dt = 1e-3
    small_dt = 1e-4
    large_amplitude = mps_overlap(
        target,
        midpoint_global_krylov_step(initial, hamiltonian, large_dt; controls...).state,
    )
    small_amplitude = mps_overlap(
        target,
        midpoint_global_krylov_step(initial, hamiltonian, small_dt; controls...).state,
    )
    large_error = large_amplitude - (-im * large_dt * direct_matrix_element)
    small_error = small_amplitude - (-im * small_dt * direct_matrix_element)

    @test mps_overlap(target, initial) ≈ 0.0 atol = 1e-14
    @test abs(direct_matrix_element) > 1e-12
    @test small_amplitude / small_dt ≈ -im * direct_matrix_element atol = 1e-9
    @test abs(small_error / small_dt) < 0.1 * abs(large_error / large_dt)
end

@testset "Global Krylov finite-time free-fermion covariance" begin
    # Lb=10 U=0 plumbing gate only. The joint endpoint/midpoint factorization
    # supplies the two midpoint MPO rows directly; no factor interpolation is
    # used. For this product Fock state, independent up/down free evolution
    # gives d_imp(t) = nup_imp(t) * ndn_imp(t).
    endpoints = [0.0, 0.01, 0.02]
    grid = midpoint_factorization_grid(endpoints)
    lesser_kernel = -im * hybridization_matrix(
        grid.points;
        component = :lesser,
        t1 = 0.25,
        intervals = 512,
    )
    greater_kernel = im * hybridization_matrix(
        grid.points;
        component = :greater,
        t1 = 0.25,
        intervals = 512,
    )
    bath = factorized_bath_spec(
        factorize_psd(lesser_kernel; rank = 5, atol = 1e-11).factor,
        factorize_psd(greater_kernel; rank = 5, atol = 1e-11).factor,
    )
    model = siam_model_spec(0.0, bath)
    sites = electron_sites(model)
    midpoint_hamiltonians = [
        siam_mpo(model, sites, index) for index in grid.midpoint_indices
    ]
    one_particle_hamiltonians = [
        free_one_particle_hamiltonian(model, index) for index in grid.midpoint_indices
    ]
    controls = (
        krylovdim = 7,
        action_cutoff = 0.0,
        action_maxdim = 1000,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 1000,
        combination_cutoff = 0.0,
        combination_maxdim = 1000,
        breakdown_tolerance = 1e-12,
    )
    exact_up = run_free_midpoint_steps(
        initial_one_particle_density(model, :Up, :Up),
        endpoints,
        one_particle_hamiltonians,
    )
    exact_down = run_free_midpoint_steps(
        initial_one_particle_density(model, :Up, :Dn),
        endpoints,
        one_particle_hamiltonians,
    )
    mps_states = MPS[initial_product_mps(model, sites, :Up)]
    for step in eachindex(midpoint_hamiltonians)
        dt = endpoints[step + 1] - endpoints[step]
        push!(
            mps_states,
            midpoint_global_krylov_step(
                mps_states[end], midpoint_hamiltonians[step], dt; controls...
            ).state,
        )
    end

    @test grid.midpoint_indices == [2, 4]
    @test length(mps_states) == length(endpoints)
    for index in eachindex(endpoints)
        mps_impurity = impurity_occupations(mps_states[index], model.impurity_position)
        up = one_particle_density_diagnostics(
            exact_up.densities[index], model.impurity_position
        ).impurity_occupation
        down = one_particle_density_diagnostics(
            exact_down.densities[index], model.impurity_position
        ).impurity_occupation
        @test mps_impurity.nup ≈ up atol = 1e-10
        @test mps_impurity.ndn ≈ down atol = 1e-10
        @test mps_impurity.double_occupancy ≈ up * down atol = 1e-10
    end
end

@testset "Global Krylov interacting spin-average plumbing" begin
    # Lb=10 U=4 technical gate only. The joint factorization grid supplies
    # midpoint rows directly; both historical atomic spin components evolve
    # independently and only scalar observables are averaged afterwards.
    endpoints = [0.0, 0.01, 0.02]
    grid = midpoint_factorization_grid(endpoints)
    lesser_kernel = -im * hybridization_matrix(
        grid.points;
        component = :lesser,
        t1 = 0.25,
        intervals = 512,
    )
    greater_kernel = im * hybridization_matrix(
        grid.points;
        component = :greater,
        t1 = 0.25,
        intervals = 512,
    )
    bath = factorized_bath_spec(
        factorize_psd(lesser_kernel; rank = 5, atol = 1e-11).factor,
        factorize_psd(greater_kernel; rank = 5, atol = 1e-11).factor,
    )
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    midpoint_hamiltonians = [
        siam_mpo(model, sites, index) for index in grid.midpoint_indices
    ]
    endpoint_hamiltonians = [
        siam_mpo(model, sites, index) for index in grid.endpoint_indices
    ]
    controls = (
        krylovdim = 7,
        action_cutoff = 0.0,
        action_maxdim = 1000,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 1000,
        combination_cutoff = 0.0,
        combination_maxdim = 1000,
        breakdown_tolerance = 1e-12,
    )

    up_result = run_global_krylov_component(
        initial_product_mps(model, sites, :Up),
        endpoints,
        midpoint_hamiltonians,
        endpoint_hamiltonians,
        model.impurity_position;
        controls...,
    )
    down_result = run_global_krylov_component(
        initial_product_mps(model, sites, :Dn),
        endpoints,
        midpoint_hamiltonians,
        endpoint_hamiltonians,
        model.impurity_position;
        controls...,
    )
    averaged_records = average_spin_component_records(up_result.records, down_result.records)

    @test grid.midpoint_indices == [2, 4]
    @test length(up_result.states) == length(endpoints)
    @test length(down_result.states) == length(endpoints)
    @test all(diagnostics -> isapprox(diagnostics.output_norm, 1.0; atol = 1e-9), up_result.step_diagnostics)
    @test all(diagnostics -> isapprox(diagnostics.output_norm, 1.0; atol = 1e-9), down_result.step_diagnostics)
    @test all(diagnostics -> diagnostics.projected_residual ≥ 0.0, up_result.step_diagnostics)
    @test all(diagnostics -> diagnostics.projected_residual ≥ 0.0, down_result.step_diagnostics)
    @test isempty(validate_trajectory_records(averaged_records))
    @test [record.time for record in averaged_records] == endpoints
    for index in eachindex(endpoints)
        up = up_result.records[index].diagnostics
        down = down_result.records[index].diagnostics
        average = averaged_records[index].diagnostics
        @test up.norm ≈ 1.0 atol = 1e-9
        @test down.norm ≈ 1.0 atol = 1e-9
        @test up.total_particle_number ≈ 11.0 atol = 1e-9
        @test down.total_particle_number ≈ 11.0 atol = 1e-9
        @test up.spin_projection ≈ 0.5 atol = 1e-9
        @test down.spin_projection ≈ -0.5 atol = 1e-9
        @test up.impurity_nup ≈ down.impurity_ndn atol = 1e-9
        @test up.impurity_ndn ≈ down.impurity_nup atol = 1e-9
        @test up.impurity_double_occupancy ≈ down.impurity_double_occupancy atol = 1e-9
        @test spin_average(up.impurity_nup, down.impurity_nup) ≈
              spin_average(up.impurity_ndn, down.impurity_ndn) atol = 1e-9
        @test average.norm ≈ spin_average(up.norm, down.norm) atol = 1e-12
        @test average.total_particle_number ≈ 11.0 atol = 1e-9
        @test average.impurity_nup ≈ average.impurity_ndn atol = 1e-9
        @test average.spin_projection ≈ 0.0 atol = 1e-9
        @test average.max_link_dimension == max(up.max_link_dimension, down.max_link_dimension)
        @test 0.0 ≤ average.impurity_double_occupancy ≤ 1.0
    end
    @test_throws ArgumentError average_spin_component_records(
        up_result.records, down_result.records[1:end-1]
    )
    @test_throws ArgumentError average_spin_component_record(
        up_result.records[1], down_result.records[2]
    )
    @test_throws ArgumentError run_global_krylov_component(
        initial_product_mps(model, sites, :Up),
        [0.0],
        midpoint_hamiltonians,
        endpoint_hamiltonians,
        model.impurity_position;
        controls...,
    )
end

@testset "Global Krylov numerical convergence" begin
    # Static Lb=10 U=0 bath: every refinement evolves the identical finite
    # Hamiltonian, so this isolates propagation controls from bath fitting.
    # This is a technical acceptance gate, not a Wolf trajectory or a bath
    # convergence claim.
    bath = factorized_bath_spec(
        fill(0.2 + 0.1im, 1, 5),
        fill(-0.1 + 0.05im, 1, 5),
    )
    model = siam_model_spec(0.0, bath)
    sites = electron_sites(model)
    hamiltonian = siam_mpo(model, sites, 1)
    one_particle_hamiltonian = free_one_particle_hamiltonian(model, 1)
    initial_up = initial_one_particle_density(model, :Up, :Up)
    initial_down = initial_one_particle_density(model, :Up, :Dn)
    exact_up = free_midpoint_step(initial_up, one_particle_hamiltonian, 0.02)
    exact_down = free_midpoint_step(initial_down, one_particle_hamiltonian, 0.02)
    exact_observables = (
        nup = one_particle_density_diagnostics(
            exact_up, model.impurity_position
        ).impurity_occupation,
        ndn = one_particle_density_diagnostics(
            exact_down, model.impurity_position
        ).impurity_occupation,
    )
    exact_double_occupancy = exact_observables.nup * exact_observables.ndn
    reference_controls = (
        krylovdim = 8,
        action_cutoff = 0.0,
        action_maxdim = 1000,
        orthogonalization_cutoff = 0.0,
        orthogonalization_maxdim = 1000,
        combination_cutoff = 0.0,
        combination_maxdim = 1000,
        breakdown_tolerance = 1e-12,
    )

    function evolve_static_global_krylov(controls, step_count)
        state = initial_product_mps(model, sites, :Up)
        dt = 0.02 / step_count
        for _ in 1:step_count
            state = midpoint_global_krylov_step(state, hamiltonian, dt; controls...).state
        end
        occupations = impurity_occupations(state, model.impurity_position)
        return (
            nup = occupations.nup,
            ndn = occupations.ndn,
            double_occupancy = occupations.double_occupancy,
        )
    end

    function observable_error(observables)
        return maximum(
            abs.(
                (
                    observables.nup - exact_observables.nup,
                    observables.ndn - exact_observables.ndn,
                    observables.double_occupancy - exact_double_occupancy,
                ),
            ),
        )
    end

    # The fine reference independently tightens timestep and Krylov controls.
    # Compression variants use the coarser single step to keep this Lb=10
    # technical gate bounded while changing exactly one control family.
    fine = evolve_static_global_krylov(reference_controls, 2)
    coarse = evolve_static_global_krylov(reference_controls, 1)
    lower_krylov = evolve_static_global_krylov(merge(reference_controls, (krylovdim = 5,)), 2)
    tighter_action = evolve_static_global_krylov(
        merge(reference_controls, (action_maxdim = 100,)), 1
    )
    tighter_orthogonalization = evolve_static_global_krylov(
        merge(reference_controls, (orthogonalization_maxdim = 100,)), 1
    )
    tighter_combination = evolve_static_global_krylov(
        merge(reference_controls, (combination_maxdim = 100,)), 1
    )

    fine_error = observable_error(fine)
    coarse_error = observable_error(coarse)
    lower_krylov_error = observable_error(lower_krylov)
    action_error = observable_error(tighter_action)
    orthogonalization_error = observable_error(tighter_orthogonalization)
    combination_error = observable_error(tighter_combination)

    @test fine_error ≤ 1e-10
    @test fine_error ≤ coarse_error + 1e-12
    @test fine_error ≤ lower_krylov_error + 1e-12
    @test action_error ≤ 1e-10
    @test orthogonalization_error ≤ 1e-10
    @test combination_error ≤ 1e-10
end

@testset "Atomic initial states and local observables" begin
    # Static Lb=10 product states only; no time evolution is performed here.
    lesser_factor = fill(0.1 + 0.2im, 2, 5)
    greater_factor = fill(-0.2 + 0.1im, 2, 5)
    bath = factorized_bath_spec(lesser_factor, greater_factor)
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)

    up_state = initial_product_mps(model, sites, :Up)
    down_state = initial_product_mps(model, sites, :Dn)
    @test length(up_state) == length(sites)
    @test length(down_state) == length(sites)
    @test_throws ArgumentError initial_product_mps(model, sites[1:10], :Up)
    @test_throws ArgumentError initial_product_mps(model, sites, :Emp)

    up = impurity_occupations(up_state, model.impurity_position)
    down = impurity_occupations(down_state, model.impurity_position)
    @test up.nup ≈ 1.0 atol = 1e-12
    @test up.ndn ≈ 0.0 atol = 1e-12
    @test up.double_occupancy ≈ 0.0 atol = 1e-12
    @test down.nup ≈ 0.0 atol = 1e-12
    @test down.ndn ≈ 1.0 atol = 1e-12
    @test down.double_occupancy ≈ 0.0 atol = 1e-12
    @test impurity_double_occupancy(up_state, model.impurity_position) ≈ 0.0 atol = 1e-12
    diagnostics = state_diagnostics(up_state, siam_mpo(model, sites, 1), model.impurity_position)
    @test diagnostics.norm ≈ 1.0 atol = 1e-12
    @test diagnostics.total_particle_number ≈ 11.0 atol = 1e-12
    @test diagnostics.spin_projection ≈ 0.5 atol = 1e-12
    @test diagnostics.energy ≈ -1.0 atol = 1e-12
    @test diagnostics.energy_imaginary_part ≈ 0.0 atol = 1e-12
    @test diagnostics.impurity_double_occupancy ≈ 0.0 atol = 1e-12
    @test diagnostics.max_link_dimension == 1
    @test all(==(1), diagnostics.link_dimensions)
    @test spin_average(up.nup, down.nup) ≈ 0.5 atol = 1e-12
    @test spin_average(up.ndn, down.ndn) ≈ 0.5 atol = 1e-12
    @test spin_average(up.double_occupancy, down.double_occupancy) ≈ 0.0 atol = 1e-12
    @test_throws ArgumentError impurity_occupations(up_state, 0)
    @test_throws ArgumentError impurity_double_occupancy(up_state, length(up_state) + 1)
end

@testset "Midpoint-inclusive factorization grid" begin
    grid = midpoint_factorization_grid([0.0, 0.1, 0.3])
    @test grid.endpoints == [0.0, 0.1, 0.3]
    @test grid.midpoints == [0.05, 0.2]
    @test grid.points == [0.0, 0.05, 0.1, 0.2, 0.3]
    @test grid.endpoint_indices == [1, 3, 5]
    @test grid.midpoint_indices == [2, 4]

    lesser = -im * hybridization_matrix(
        grid.points;
        component = :lesser,
        t1 = 0.25,
        intervals = 512,
    )
    greater = im * hybridization_matrix(
        grid.points;
        component = :greater,
        t1 = 0.25,
        intervals = 512,
    )
    lesser_factor = factorize_psd(lesser; rank = 5, atol = 1e-11).factor
    greater_factor = factorize_psd(greater; rank = 5, atol = 1e-11).factor
    @test factor_rows(lesser_factor, grid.midpoint_indices) ==
          lesser_factor[grid.midpoint_indices, :]
    @test factor_rows(lesser_factor, grid.midpoint_indices) !==
          lesser_factor[grid.midpoint_indices, :]
    rotated_factor = copy(lesser_factor)
    rotated_factor[:, 1] .*= im
    midpoint_factor = factor_rows(lesser_factor, grid.midpoint_indices)
    rotated_midpoint_factor = factor_rows(rotated_factor, grid.midpoint_indices)
    @test rotated_midpoint_factor * rotated_midpoint_factor' ≈
          midpoint_factor * midpoint_factor' atol = 1e-12

    # Midpoint rows come from the one joint factorization, not interpolation.
    bath = factorized_bath_spec(lesser_factor, greater_factor)
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    midpoint_hamiltonian = siam_mpo(model, sites, grid.midpoint_indices[1])
    @test length(midpoint_hamiltonian) == 11

    @test_throws ArgumentError midpoint_factorization_grid([0.0])
    @test_throws ArgumentError midpoint_factorization_grid([0.0, 0.0])
    @test_throws ArgumentError midpoint_factorization_grid([0.1, 0.0])
    @test_throws ArgumentError midpoint_factorization_grid([0.0, Inf])
    @test_throws BoundsError factor_rows(lesser_factor, [0])
end

@testset "Versioned trajectory output schema" begin
    records = [
        (
            time = 0.0,
            diagnostics = (
                norm = 1.0,
                total_particle_number = 11.0,
                spin_projection = 0.5,
                energy = -1.0,
                energy_imaginary_part = 0.0,
                impurity_nup = 1.0,
                impurity_ndn = 0.0,
                impurity_double_occupancy = 0.0,
                max_link_dimension = 1,
            ),
        ),
        (
            time = 0.02,
            diagnostics = (
                norm = 1.0,
                total_particle_number = 11.0,
                spin_projection = 0.5,
                energy = -0.99,
                energy_imaginary_part = 0.0,
                impurity_nup = 0.98,
                impurity_ndn = 0.01,
                impurity_double_occupancy = 0.01,
                max_link_dimension = 4,
            ),
        ),
    ]
    @test isempty(validate_trajectory_records(records))
    table = trajectory_table(records)
    @test keys(table) == TRAJECTORY_COLUMNS
    @test table.time == [0.0, 0.02]
    @test table.max_link_dimension == [1, 4]

    scratch = mktempdir(joinpath(ROOT, "tmp"))
    trajectory_path = joinpath(scratch, "trajectory.csv")
    metadata_path = joinpath(scratch, "metadata.toml")
    @test write_trajectory_csv(trajectory_path, records) == trajectory_path
    restored = read_trajectory_csv(trajectory_path)
    @test restored.time == table.time
    @test restored.energy ≈ table.energy atol = 1e-14
    @test restored.max_link_dimension == Float64[1, 4]

    metadata = Dict(
        "schema_version" => TRAJECTORY_SCHEMA_VERSION,
        "run" => Dict("Lb" => 10, "status" => "technical-validation"),
    )
    @test write_metadata_toml(metadata_path, metadata) == metadata_path
    @test read_metadata_toml(metadata_path) == metadata
    @test_throws ArgumentError write_metadata_toml(
        metadata_path,
        Dict("schema_version" => 2),
    )
    @test_throws ArgumentError read_trajectory_csv(metadata_path)

    invalid_records = copy(records)
    invalid_records[2] = (time = 0.0, diagnostics = records[2].diagnostics)
    @test !isempty(validate_trajectory_records(invalid_records))
    @test_throws ArgumentError trajectory_table(invalid_records)
    rm(scratch; recursive = true)
end

@testset "Exact free-fermion covariance primitives" begin
    bath = factorized_bath_spec(fill(0.2 + 0.1im, 2, 5), fill(-0.1 + 0.2im, 2, 5))
    model = siam_model_spec(0.0, bath)
    hamiltonian = free_one_particle_hamiltonian(model, 1)
    up_density = initial_one_particle_density(model, :Up, :Up)
    down_density = initial_one_particle_density(model, :Up, :Dn)

    @test hamiltonian ≈ hamiltonian' atol = 1e-12
    @test up_density[model.impurity_position, model.impurity_position] == 1.0
    @test down_density[model.impurity_position, model.impurity_position] == 0.0
    @test tr(up_density) == 6.0
    @test tr(down_density) == 5.0
    propagated = free_midpoint_step(down_density, hamiltonian, 0.02)
    @test tr(propagated) ≈ 5.0 atol = 1e-12
    @test propagated ≈ propagated' atol = 1e-12
    trajectory = run_free_midpoint_steps(
        down_density,
        [0.0, 0.02, 0.04],
        [hamiltonian, hamiltonian],
    )
    @test trajectory.times == [0.0, 0.02, 0.04]
    @test length(trajectory.densities) == 3
    @test all(isapprox(tr(density), 5.0; atol = 1e-12) for density in trajectory.densities)
    @test one_particle_density_diagnostics(
        trajectory.densities[end], model.impurity_position
    ).particle_number_imaginary_part ≈ 0.0 atol = 1e-12
    @test_throws ArgumentError free_one_particle_hamiltonian(siam_model_spec(4.0, bath), 1)
    @test_throws ArgumentError initial_one_particle_density(model, :Up, :bad_spin)
    @test_throws ArgumentError free_midpoint_step(down_density, [1.0 1.0; 0.0 1.0], 0.02)
end

@testset "Wolf benchmark configuration rejects invalid bath size" begin
    config = load_benchmark_config(joinpath(ROOT, "configs", "wolf_u4.toml"))
    config["bath"]["discretizations"][1]["Lb"] = 8
    errors = validate_benchmark_config(config)
    @test any(contains("at least 10"), errors)
end
