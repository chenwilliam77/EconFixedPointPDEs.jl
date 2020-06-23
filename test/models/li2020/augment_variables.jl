# This script tests the internal functions of eqcond.jl
using Test, OrderedCollections
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))

rp = joinpath(dirname(@__FILE__), "../../reference/models/li2020")

# Set up input arguments and expected behavior
inside_output = load(joinpath(rp, "inside_iteration_loop_output.jld2"))
prepQ̂_input = load(joinpath(rp, "post_processing_input.jld2"))
aug_out = load(joinpath(rp, "augment_variables_out.jld2"), "results")

m = Li2020()
stategrid, funcvar, derivs, endo = initialize!(m)
funcvar[:p] = vec(prepQ̂_input["p_new"])
funcvar[:Q̂] = vec(prepQ̂_input["hat_Q_last"])
funcvar[:xg] = vec(prepQ̂_input["xg_new"])
endo[:κp] = vec(prepQ̂_input["kappap_vec"])
θ = parameters_to_named_tuple(m)
f = eqcond(m)
augment_variables!(m, stategrid, funcvar, derivs, endo)

@testset "Augment variables of Li2020" begin
    @test @test_matrix_approx_eq vec(aug_out["psi_vec"]) endo[:ψ]
    @test @test_matrix_approx_eq vec(aug_out["xK_vec"]) endo[:xK]
    @test @test_matrix_approx_eq vec(aug_out["Lvg"]) endo[:lvg]
    @test @test_matrix_approx_eq vec(aug_out["yK_vec"]) endo[:yK]
    @test @test_matrix_approx_eq vec(aug_out["yg_vec"]) endo[:yg]
    @test @test_matrix_approx_eq vec(aug_out["sigmap_vec"]) endo[:σp]
    @test @test_matrix_approx_eq vec(aug_out["sigmab_vec"]) endo[:σ]
    @test @test_matrix_approx_eq vec(aug_out["sigmah_vec"]) endo[:σh]
    @test @test_matrix_approx_eq vec(aug_out["firesale_jump"]) endo[:firesale_jump]
    @test @test_matrix_approx_eq vec(aug_out["kappad_vec"]) endo[:κd]
    @test @test_matrix_approx_eq vec(aug_out["kappab_vec"]) endo[:κb]
    @test @test_matrix_approx_eq vec(aug_out["kappafs_vec"]) endo[:κfs]
    @test @test_matrix_approx_eq vec(aug_out["kappah_vec"]) endo[:κh]
    @test @test_matrix_approx_eq vec(aug_out["liq_prem_vec"]) endo[:liq_prem]
    @test @test_matrix_approx_eq vec(aug_out["bank_liq_frac"]) endo[:bank_liq_frac]
    @test @test_matrix_approx_eq vec(aug_out["muR_rd_vec"]) endo[:μR_rd]
    @test @test_matrix_approx_eq vec(aug_out["mub_muh_vec"]) endo[:μb_μh]
    @test @test_matrix_approx_eq vec(aug_out["mup_vec"]) endo[:μp]
    @test @test_matrix_approx_eq vec(aug_out["muK_vec"]) endo[:μK]
    @test @test_matrix_approx_eq vec(aug_out["muR_vec"]) endo[:μR]
    @test @test_matrix_approx_eq vec(aug_out["kappaK_vec"]) endo[:κK]
    @test @test_matrix_approx_eq vec(aug_out["rd_vec"]) endo[:rd]
    @test @test_matrix_approx_eq vec(aug_out["rd_vec"]) endo[:rd]
    @test @test_matrix_approx_eq vec(aug_out["muw_vec"]) endo[:μw]
    @test @test_matrix_approx_eq vec(aug_out["sigmaw_vec"]) endo[:σw]
    @test @test_matrix_approx_eq vec(aug_out["kappaw_vec"]) endo[:κw]
    @test @test_matrix_approx_eq vec(aug_out["rg_vec"]) endo[:rg]
    @test @test_matrix_approx_eq vec(aug_out["rh_vec"]) endo[:rh]
    @test @test_matrix_approx_eq vec(aug_out["rf_vec"]) endo[:rf]
end

nothing
