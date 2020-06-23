# This script tests the internal functions of eqcond.jl
using Test, OrderedCollections, BenchmarkTools
include("../../../src/includeall.jl")

time_functions = false
rp = joinpath(dirname(@__FILE__), "../../reference/models/li2020")

# Set up input arguments and expected behavior
inside_input = load(joinpath(rp, "inside_iteration_input.jld2"))
inside_step3 = load(joinpath(rp, "inside_iteration_Step3.jld2"))
inside_output = load(joinpath(rp, "inside_iteration_loop_output.jld2"))
Q̂_input = load(joinpath(rp, "Qhat_calculation_input.jld2"))
prepQ̂_input = load(joinpath(rp, "post_processing_input.jld2"))
aug_out = load(joinpath(rp, "augment_variables_out.jld2"), "results")
Q̂_out   = load(joinpath(rp, "hat_Q_calculation_output.jld2"))

@testset "Inside Iteration" begin
    m = Li2020()
    stategrid, funcvar, derivs, endo = initialize!(m)
    θ = parameters_to_named_tuple(m)
    p₀ = aug_out["p0"]
    p₁ = aug_out["p1"]
    max_jump = (p₁ - p₀) / p₁
    p_fitted = extrapolate(interpolate((stategrid[:w], ), vec(inside_input["p_last"]), get_setting(m, :p_interpolant)), Line())
    Q = get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level)
    Φ = get_setting(m, :Φ)
    κp_grid = get_setting(m, :κp_grid)
    κp, xK, xg, a, b, c = inside_iteration_li2020(inside_input["w"], p_fitted, inside_input["p"], inside_input["pw"], inside_input["xK"],
                                                  inside_input["xg"], Q, inside_input["hat_Q"], max_jump, θ, κp_grid = κp_grid,
                                                  nlsolve_tol = 1e-6, xg_tol = 2e-3, verbose = :high)

    # First test one loop
    @test a
    @test b
    @test c
    @test κp ≈ inside_step3["kappap"] atol=1e-7
    @test xK ≈ inside_step3["xK"] atol=1e-5
    @test xg ≈ inside_step3["xg"] atol=1e-7

    # Now test the whole loop
    p = vec(inside_input["p_last"])
    Q̂ = fill(inside_input["hat_Q"], length(p))
    xg = Q ./ (2. .* stategrid[:w] .* p)
    xg[1] = 2 * xg[2]
    xg[end] = θ[:ϵ]
    ∂p∂w = differentiate(stategrid[:w], p)
    ψ  = map((x, y) -> (Φ(x, θ[:χ], θ[:δ]) + θ[:ρ] * (x + y) - θ[:AL]) / (θ[:AH] - θ[:AL]), p, Q̂)
    xK = ψ ./ stategrid[:w] .* p ./ (p + Q̂)

    solved_κp = BitArray(undef, length(p))
    solved_xK = BitArray(undef, length(p))
    solved_xg = BitArray(undef, length(p))
    solved_xK[1] = true
    solved_κp[1] = true
    solved_xg[1] = true
    solved_xK[end] = true
    solved_κp[end] = true
    solved_xg[end] = true
    κp_new = zeros(length(p))
    xg_new = deepcopy(xg)
    p_new = deepcopy(p)
    for i in 2:(length(stategrid) - 1)
        wᵢ    = stategrid[:w][i]
        pᵢ    = p[i]
        xKᵢ   = xK[i]
        xgᵢ   = xg[i]
        ∂p∂wᵢ = ∂p∂w[i]
        Q̂ᵢ    = Q̂[i]

        κp_new[i], xKᵢ, xg_new[i], succeed_κp, succeed_xK, succeed_xg =
            inside_iteration_li2020(wᵢ, p_fitted, pᵢ, ∂p∂wᵢ, xKᵢ, xgᵢ, Q, Q̂ᵢ, max_jump, θ;
                                    κp_grid = κp_grid, nlsolve_tol = 1e-6, xg_tol = 2e-3, nlsolve_iter = 400)

        ψᵢ = xKᵢ * wᵢ / (pᵢ / (pᵢ + Q̂ᵢ))
        p_new[i] = nlsolve(x -> Φ(x[1], θ[:χ], θ[:δ]) .+ θ[:ρ] .* (x .+ Q̂ᵢ) .- (ψᵢ * θ[:AH] + (1. - ψᵢ) * θ[:AL]),
                           [(p₀ + p₁) / 2.]).zero[1]
        solved_κp[i] = succeed_κp
        solved_xK[i] = succeed_xK
        solved_xg[i] = succeed_xg
    end

    problem_inds = [11, 14, 15, 16, 21, 97]
    nonproblem_inds = setdiff(1:length(p), problem_inds)
    @test vec(inside_output["xg_new"])[nonproblem_inds] ≈ xg_new[nonproblem_inds] atol=9e-5
    @test abs(inside_output["xg_new"][11] - xg_new[11]) < 1.32
    @test abs(inside_output["xg_new"][14] - xg_new[14]) < 0.6
    @test abs(inside_output["xg_new"][15] - xg_new[15]) < 0.31
    @test abs(inside_output["xg_new"][16] - xg_new[16]) < 0.03
    @test abs(inside_output["xg_new"][21] - xg_new[21]) < 2e-4
    @test abs(inside_output["xg_new"][97] - xg_new[97]) < 2e-3
    @test vec(inside_output["p_new"]) ≈ p_new atol=8e-5
    @test vec(inside_output["kappap_vec"]) ≈ κp_new atol=5.5e-4
    @test all(solved_κp)
    @test all(solved_xK)
    @test all(solved_xg)
end

@testset "Prepare for and calculate Q̂" begin
    m = Li2020()
    stategrid, funcvar, derivs, endo = initialize!(m)
    θ = parameters_to_named_tuple(m)
    f_μK = get_setting(m, :μK)
    Φ = get_setting(m, :Φ)
    yg_tol = get_setting(m, :yg_tol)
    firesale_bound = get_setting(m, :firesale_bound)
    firesale_interpolant = get_setting(m, :firesale_interpolant)
    Q = get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level)
    funcvar[:p] = vec(prepQ̂_input["p_new"])
    funcvar[:Q̂] = vec(prepQ̂_input["hat_Q_last"])
    funcvar[:xg] = vec(prepQ̂_input["xg_new"])
    endo[:κp] = vec(prepQ̂_input["kappap_vec"])
    prepare_Q̂!(stategrid, funcvar, derivs, endo, θ, f_μK, Φ, yg_tol, firesale_bound, firesale_interpolant, Q)

    @test @test_matrix_approx_eq vec(aug_out["muw_vec"]) endo[:μw]
    @test @test_matrix_approx_eq vec(aug_out["sigmaw_vec"]) endo[:σw]
    @test @test_matrix_approx_eq vec(aug_out["kappaw_vec"]) endo[:κw]
    @test @test_matrix_approx_eq vec(aug_out["rg_vec"]) endo[:rg]
    @test @test_matrix_approx_eq vec(aug_out["rh_vec"]) endo[:rh]
    @test @test_matrix_approx_eq vec(aug_out["rf_vec"]) endo[:rf]

    if time_functions
        BenchmarkTools.@btime begin
            Q̂_calculation!(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
                           vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
                           Q̂_input["Qval"], Q̂_input["lambda"])
        end
    end

    my_Q̂ = Q̂_calculation!(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
                          vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
                          Q̂_input["Qval"], Q̂_input["lambda"])
    @test maximum(abs.(my_Q̂ - vec(Q̂_out["hat_Q_vec"]))) < 5e-4
end

@testset "eqcond" begin
    m = Li2020()
    Q = get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level)
    stategrid, funcvar, derivs, endo = initialize!(m)
    bc = get_setting(m, :boundary_conditions)
    bc[:p] .= [aug_out["p0"], aug_out["p1"]] # Change boundary conditions to match MATLAB output
    indiv_convergence = OrderedDict{Symbol, Vector{eltype(m)}}(:Q̂ => [0., 1e-3])
    p = vec(inside_input["p_last"])
    Q̂ = fill(inside_input["hat_Q"], length(p))
    θ = parameters_to_named_tuple(m)
    xg = Q ./ (2. .* stategrid[:w] .* p)
    xg[1] = 2 * xg[2]
    xg[end] = θ[:ϵ]
    funcvar[:p] .= p
    funcvar[:xg] .= xg
    funcvar[:Q̂] .= Q̂
    θ = parameters_to_named_tuple(m)
    f = eqcond(m)
    @info "The following output is expected; checking high verbosity statements are shown."
    p_new, xg_new, Q̂_new = f(stategrid, funcvar, derivs, endo, θ; individual_convergence = indiv_convergence,
                             verbose = :high)
    # Very rough bounds
    @test p_new ≈ aug_out["p_new"] atol=1e-2
    @test p_new[1:41] ≈ aug_out["p_new"][1:41] atol=2e-3
    @test p_new[42:48] ≈ aug_out["p_new"][42:48] atol=6e-3
    @test p_new[49:end - 1] ≈ aug_out["p_new"][49:end - 1] atol=1e-6
    @test p_new[end] == p_new[end - 1]
    @test Q̂_new ≈ Q̂_out["hat_Q_vec"] atol=1e-2
    @test xg_new[1:12] ≈ inside_output["xg_new"][1:12]
    @test xg_new[13:19] ≈ inside_output["xg_new"][13:19] atol=1.5
    @test xg_new[20:30] ≈ inside_output["xg_new"][20:30] atol=8e-2
    @test xg_new[30:end] ≈ inside_output["xg_new"][30:end] atol=5e-3

    if time_functions
        BenchmarkTools.@btime begin # roughly 3s vs. 30s of MATLAB (2.8s if not starting from guess of zeros for Q̂)
            p = vec(inside_input["p_last"])
            Q̂ = fill(inside_input["hat_Q"], length(p))
            θ = parameters_to_named_tuple(m)
            xg = Q ./ (2. .* stategrid[:w] .* p)
            xg[1] = 2 * xg[2]
            xg[end] = θ[:ϵ]
            funcvar[:p] .= p
            funcvar[:xg] .= xg
            funcvar[:Q̂] .= Q̂
            indiv_convergence = OrderedDict{Symbol, Vector{eltype(m)}}(:Q̂ => [0., 1e-3])
            f(stategrid, funcvar, derivs, endo, θ; individual_convergence = indiv_convergence)
        end
    end
end

nothing
