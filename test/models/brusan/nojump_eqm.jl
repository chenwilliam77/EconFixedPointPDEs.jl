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

@info "The following warning is expected."
stategrid, funcvar, derivs, endo = solve(m; nojump = true)

if regen_output
    h5open(joinpath(dirname(@__FILE__), "../../reference/models/brusan/brusan_nojump_eqm_log.h5"), "w") do file
        write(file, "q_Tsit5", sol.u)
        write(file, "q_Tsit5_solve", funcvar[:q])
        for (k, v) in endo
            write(file, "$(detexify(k))_Tsit5_solve", v)
        end
    end
end

true_soln = h5read(joinpath(dirname(@__FILE__), "../../reference/models/brusan/brusan_nojump_eqm_log.h5"), "q_Tsit5")
@testset "Solution for no jump equilibrium, log utility" begin
    @test @test_matrix_approx_eq sol.u true_soln

    file = joinpath(dirname(@__FILE__), "../../reference/models/brusan/brusan_nojump_eqm_log.h5")
    for (k, v) in endo
        @test @test_matrix_approx_eq h5read(file, "$(detexify(k))_Tsit5_solve") v
    end
end
