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

@testset "One-step midpoint TDVP" begin
    # An Lb=10 technical propagation check only, not a physics trajectory.
    # The factor rows are deliberately supplied directly to the model; this
    # test makes no implicit midpoint interpolation choice.
    lesser_factor = fill(0.15 + 0.1im, 2, 5)
    greater_factor = fill(-0.1 + 0.05im, 2, 5)
    bath = factorized_bath_spec(lesser_factor, greater_factor)
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    state = initial_product_mps(model, sites, :Up)
    midpoint_hamiltonian = siam_mpo(model, sites, 1)

    unchanged = midpoint_tdvp_step(state, midpoint_hamiltonian, 0.0)
    @test unchanged !== state
    @test unchanged ≈ state atol = 1e-14

    evolved = midpoint_tdvp_step(
        state,
        midpoint_hamiltonian,
        0.02;
        cutoff = 1e-12,
        maxdim = 100,
        nsite = 2,
        outputlevel = 0,
    )
    @test norm(evolved) ≈ 1.0 atol = 1e-10
    @test sum(expect(evolved, "Ntot")) ≈ 11.0 atol = 1e-9
    @test sum(expect(evolved, "Nup")) - sum(expect(evolved, "Ndn")) ≈ 1.0 atol = 1e-9
    @test impurity_double_occupancy(evolved, model.impurity_position) ≥ -1e-12
    @test impurity_double_occupancy(evolved, model.impurity_position) ≤ 1.0 + 1e-12

    observed = midpoint_tdvp_observed_step(
        state,
        midpoint_hamiltonian,
        0.02,
        model.impurity_position;
        cutoff = 1e-12,
        maxdim = 100,
        nsite = 2,
        outputlevel = 0,
    )
    @test observed.dt == 0.02
    @test norm(observed.state) ≈ 1.0 atol = 1e-10
    @test observed.diagnostics.norm ≈ 1.0 atol = 1e-10
    @test observed.diagnostics.total_particle_number ≈ 11.0 atol = 1e-9
    @test observed.diagnostics.spin_projection ≈ 0.5 atol = 1e-9
    @test abs(observed.diagnostics.energy_imaginary_part) ≤ 1e-9
    @test 0.0 ≤ observed.diagnostics.impurity_double_occupancy ≤ 1.0
    @test observed.diagnostics.max_link_dimension ≥ 1

    @test_throws ArgumentError midpoint_tdvp_step(state, midpoint_hamiltonian, Inf)
    @test_throws ArgumentError midpoint_tdvp_step(state, midpoint_hamiltonian, 0.02; cutoff = -1e-12)
    @test_throws ArgumentError midpoint_tdvp_step(state, midpoint_hamiltonian, 0.02; maxdim = 0)
    @test_throws ArgumentError midpoint_tdvp_step(state, midpoint_hamiltonian, 0.02; nsite = 3)
end

@testset "Bounded midpoint trajectory composition" begin
    # Two Lb=10 technical steps. This validates composition and records, not
    # a physical benchmark trajectory.
    lesser_factor = fill(0.15 + 0.1im, 3, 5)
    greater_factor = fill(-0.1 + 0.05im, 3, 5)
    bath = factorized_bath_spec(lesser_factor, greater_factor)
    model = siam_model_spec(4.0, bath)
    sites = electron_sites(model)
    state = initial_product_mps(model, sites, :Up)
    hamiltonians = [siam_mpo(model, sites, index) for index in 1:3]
    endpoints = [0.0, 0.02, 0.04]

    step = midpoint_trajectory_step(
        state,
        endpoints[1],
        endpoints[2],
        hamiltonians[1],
        hamiltonians[2],
        model.impurity_position;
        cutoff = 1e-12,
        maxdim = 100,
        outputlevel = 0,
    )
    @test step.record.time == endpoints[2]
    @test step.record.diagnostics.norm ≈ 1.0 atol = 1e-10
    @test step.record.diagnostics.total_particle_number ≈ 11.0 atol = 1e-9

    result = run_midpoint_steps(
        state,
        endpoints,
        hamiltonians[1:2],
        hamiltonians,
        model.impurity_position;
        cutoff = 1e-12,
        maxdim = 100,
        outputlevel = 0,
    )
    @test result.state !== state
    @test result.times == endpoints
    @test length(result.records) == length(endpoints)
    @test [record.time for record in result.records] == endpoints
    @test all(record -> isapprox(record.diagnostics.norm, 1.0; atol = 1e-10), result.records)
    @test all(
        record -> isapprox(record.diagnostics.total_particle_number, 11.0; atol = 1e-9),
        result.records,
    )
    @test all(
        record -> isapprox(record.diagnostics.spin_projection, 0.5; atol = 1e-9),
        result.records,
    )
    @test all(record -> 0.0 <= record.diagnostics.impurity_double_occupancy <= 1.0, result.records)

    @test_throws ArgumentError endpoint_record(state, Inf, hamiltonians[1], model.impurity_position)
    @test_throws ArgumentError midpoint_trajectory_step(
        state,
        endpoints[2],
        endpoints[1],
        hamiltonians[1],
        hamiltonians[2],
        model.impurity_position,
    )
    @test_throws ArgumentError run_midpoint_steps(
        state,
        [0.0, 0.02],
        hamiltonians[1:2],
        hamiltonians,
        model.impurity_position,
    )
    @test_throws ArgumentError run_midpoint_steps(
        state,
        [0.0, 0.02, 0.01],
        hamiltonians[1:2],
        hamiltonians,
        model.impurity_position,
    )
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

@testset "Wolf benchmark configuration rejects invalid bath size" begin
    config = load_benchmark_config(joinpath(ROOT, "configs", "wolf_u4.toml"))
    config["bath"]["discretizations"][1]["Lb"] = 8
    errors = validate_benchmark_config(config)
    @test any(contains("at least 10"), errors)
end
