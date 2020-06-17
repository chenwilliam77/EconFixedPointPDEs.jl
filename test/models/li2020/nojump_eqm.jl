using Test, OrdinaryDiffEq, HDF5, ModelConstructors
include("../../../src/includeall.jl")

m = Li2020()
stategrid, _ = initialize!(m)
ode_f, ode_callback = eqcond_nojump(m)
tspan = (stategrid[:w][1], stategrid[:w][end])
θ = parameters_to_named_tuple(map(x -> m.parameters[m.keys[x]], get_setting(m, :nojump_parameters)))
prob = ODEProblem(ode_f, get_setting(m, :boundary_conditions)[:p][1], tspan, θ, tstops = stategrid[:w][2:end - 1])
sol = solve(prob, get_setting(m, :ode_integrator),
            reltol = get_setting(m, :ode_reltol),
            abstol = get_setting(m, :ode_abstol), callback = ode_callback)

true_soln = h5read("../../reference/models/li2020/li2020_nojump_eqm.h5", "p_DP5")
@testset "Baseline solution for no jump equilibrium" begin
    @test @test_matrix_approx_eq sol.u true_soln
end
