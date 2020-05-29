using Test, DifferentialEquations
include("../../../src/includeall.jl")

m = Li2020()
stategrid, endogenous_variables = initialize(m)
ode_f = eqcond_nojump(m)
tspan = (stategrid[:w][1], stategrid[:w][end])
prob = ODEProblem(ode_f, get_setting(m, :boundary_conditions)[:p][1], tspan)
sol = solve(prob, get_setting(m, :ode_integrator),
            reltol = get_setting(m, :ode_reltol),
            abstol = get_setting(m, :ode_abstol))
