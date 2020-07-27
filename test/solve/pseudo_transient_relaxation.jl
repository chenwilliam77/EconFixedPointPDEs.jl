using Test, OrderedCollections, HDF5, BenchmarkTools
include(joinpath(dirname(@__FILE__), "../../src/includeall.jl"))

time_results = false

@testset "Pseudo-Transient Relaxation" begin
    rp = joinpath(dirname(@__FILE__), "../reference/solve/pseudo_transient_relaxation.h5")
    η = h5read(rp, "eta")
    vₑ = h5read(rp, "ve")
    vₑ₁ = h5read(rp, "ve1")
    vₕ = h5read(rp, "vh")
    vₕ₁ = h5read(rp, "vh1")
    Mvₑ = h5read(rp, "Mve")
    Mvₕ = h5read(rp, "Mvh")
    MU = h5read(rp, "MU")
    S = h5read(rp, "S")
    Δ = h5read(rp, "Delta")
    G = h5read(rp, "G")
    stategrid = StateGrid(OrderedDict(:η => η))

    # Test copy from Princeton Initiative script
    myvₑ₁ = upwind_parabolic_pde(η, Mvₑ, MU, S.^2, G, vₑ, Δ)
    myvₕ₁ = upwind_parabolic_pde(η, Mvₕ, MU, S.^2, G, vₕ, Δ)
    @test myvₑ₁ ≈ vₑ₁
    @test myvₕ₁ ≈ vₕ₁

    # Test against an implementation where the operators are constructed rather than the raw LHS matrix
    N = length(η)
    DU1 = zeros(N)
    DU2 = zeros(N)
    DD1 = zeros(N)
    DD2 = zeros(N)
    D01 = zeros(N)
    D02 = zeros(N)
    Σ²0 = zeros(N)
    dX = diff(η)
    Σ²0[2:N-1] .= S[2:N-1].^2 ./ (dX[1:N-2] + dX[2:N-1]) # approx Σ² / (2 * dx): this term is the Σ²/2 coefficient
    DU1[2:N] = max.(MU[1:N - 1], 0.) ./ dX
    DU2[2:N] = Σ²0[1:N - 1] ./ dX
    DD1[1:N - 1] = max.(-MU[2:N], 0.) ./ dX
    DD2[1:N - 1] = Σ²0[2:N] ./ dX
    D01[1:N - 1] .= -DU1[2:N]
    D01[2:N] .-= DD1[1:N - 1]
    D02[1:N - 1] .= -DU2[2:N]
    D02[2:N] .-= DD2[1:N - 1]
    L1 = spdiagm(0 => D01, -1 => DD1[1:N - 1], 1 => DU1[2:N])
    L2 = spdiagm(0 => D02, -1 => DD2[1:N - 1], 1 => DU2[2:N])
    A = spdiagm(0 => (1 - Δ) .* ones(N) + Δ .* Mvₑ) - Δ .* (L1 + L2)
    directvₑ₁ = A \ ((1 - Δ) .* vₑ)
    @test myvₑ₁ ≈ directvₑ₁

    # Test new wrapper scripts utilizing DiffEqOperator for more flexible upwinding (e.g. higher-order)
    L₁ = UpwindDifference(1, 1, diff(η)[1], N)
    L₂ = CenteredDifference(2, 2, diff(η)[1], N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), diff(η)[1])
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                 MU ./ η, (S ./ η).^2, Δ; uniform = true,
                                 L₁ = L₁, L₂ = L₂, Q = Qbc)

    @test myvₑ₁ ≈ vₑ₁
    @test myvₕ₁ ≈ vₕ₁

    L₁ = UpwindDifference(1, 1, diff(η)[1], N)
    L₂ = CenteredDifference(2, 2, diff(η)[1], N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), diff(η)[1])
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                 MU ./ η, (S ./ η).^2, Δ; uniform = false,
                                 L₁ = L₁, L₂ = L₂, Q = Qbc)

    @test myvₑ₁ ≈ vₑ₁
    @test myvₕ₁ ≈ vₕ₁

    # Test fast default using BandedMatrices
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                 MU ./ η, (S ./ η).^2, Δ; uniform = true)

    @test @test_matrix_approx_eq myvₑ₁[2:end - 1] vₑ₁[2:end - 1]
    @test @test_matrix_approx_eq myvₕ₁ vₕ₁

    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                 MU ./ η, (S ./ η).^2, Δ; uniform = false)

    @test @test_matrix_approx_eq myvₑ₁ vₑ₁
    @test @test_matrix_approx_eq myvₕ₁ vₕ₁
end

if time_results
    println("\nTiming different methods for calculating the implicit time step.\n")
    rp = joinpath(dirname(@__FILE__), "../reference/solve/pseudo_transient_relaxation.h5")
    η = h5read(rp, "eta")
    vₑ = h5read(rp, "ve")
    vₑ₁ = h5read(rp, "ve1")
    vₕ = h5read(rp, "vh")
    vₕ₁ = h5read(rp, "vh1")
    Mvₑ = h5read(rp, "Mve")
    Mvₕ = h5read(rp, "Mvh")
    MU = h5read(rp, "MU")
    S = h5read(rp, "S")
    Δ = h5read(rp, "Delta")
    G = h5read(rp, "G")

    # Test copy from Princeton Initiative script
    println("Princeton Initiative script")
    @btime begin
        upwind_parabolic_pde(η, Mvₑ, MU, S.^2, G, vₑ, Δ)
        upwind_parabolic_pde(η, Mvₕ, MU, S.^2, G, vₕ, Δ)
    end

    # Test new wrapper scripts utilizing DiffEqOperator for more flexible upwinding (e.g. higher-order)
    L₁ = UpwindDifference(1, 1, diff(η)[1], N)
    L₂ = CenteredDifference(2, 2, diff(η)[1], N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), diff(η)[1])
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    println("Using DiffEqOperator finite differencing, uniform")
    @btime begin
        pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ; uniform = true,
                                     L₁ = L₁, L₂ = L₂, Q = Qbc)
    end

    L₁ = UpwindDifference(1, 1, diff(η)[1], N)
    L₂ = CenteredDifference(2, 2, diff(η)[1], N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), diff(η)[1])
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    println("Using DiffEqOperator finite differencing, non-uniform")
    @btime begin
        pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ; uniform = false,
                                     L₁ = L₁, L₂ = L₂, Q = Qbc)
    end

    # Test fast default using BandedMatrices
    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    println("Using BandedMatrices, uniform")
    @btime begin
        pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ; uniform = true)
    end

    myvₑ₁ = similar(vₑ)
    myvₕ₁ = similar(vₕ)
    println("Using BandedMatrices, non-uniform")
    @btime begin
        pseudo_transient_relaxation!(stategrid, (myvₑ₁, myvₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ; uniform = false)
    end
end

nothing
