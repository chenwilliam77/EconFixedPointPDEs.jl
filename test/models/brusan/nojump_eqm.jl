using Test, OrdinaryDiffEq, HDF5, ModelConstructors
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))

regen_output = false

m = BruSan()
stategrid, _ = initialize_nojump!(m)
ode_f, ode_callback = eqcond_nojump(m)
tspan = (stategrid[:η][1], stategrid[:η][end])
θ = parameters_to_named_tuple(m.parameters)
prob = ODEProblem(ode_f, get_setting(m, :boundary_conditions)[:q][1], tspan, θ, tstops = stategrid[:η][2:end - 1])
sol = solve(prob, get_setting(m, :ode_integrator),
            reltol = get_setting(m, :ode_reltol),
            abstol = get_setting(m, :ode_abstol), callback = ode_callback)

if regen_output
    h5open(joinpath(dirname(@__FILE__), "../../reference/models/brusan/brusan_nojump_eqm_log.h5"), "w") do file
        write(file, "q_Tsit5", sol.u)
    end
end

true_soln = h5read(joinpath(dirname(@__FILE__), "../../reference/models/brusan/brusan_nojump_eqm_log.h5"), "q_Tsit5")
@testset "Baseline solution for no jump equilibrium" begin
    @test @test_matrix_approx_eq sol.u true_soln
end
