# This script tests the internal functions of eqcond.jl
using Test, OrderedCollections
using BenchmarkTools
# include("../../../src/includeall.jl")

rp = joinpath(dirname(@__FILE__), "../../reference/models/li2020")

# Set up input arguments and expected behavior
inside_input = load(joinpath(rp, "inside_iteration_input.jld2"))
Q̂_input = load(joinpath(rp, "Qhat_calculation_input.jld2"))
prepQ̂_input = load(joinpath(rp, "post_processing_input.jld2"))
aug_out = load(joinpath(rp, "augment_variables_out.jld2"), "results")
Q̂_out   = load(joinpath(rp, "hat_Q_calculation_output.jld2"))

@testset "Prepare variables for Q̂" begin
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
end
#=
@btime begin
    Q̂_calculation(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
              vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
              Q̂_input["Qval"], Q̂_input["lambda"])
end
=#

@testset "Calculate Q̂" begin
    my_Q̂ = Q̂_calculation(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
                         vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
                         Q̂_input["Qval"], Q̂_input["lambda"])
    @test maximum(abs.(my_Q̂ - vec(Q̂_out["hat_Q_vec"]))) < 5e-4
end
