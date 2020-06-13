"""
```
solve(m; nojump = false)
```
"""
function solve(m::AbstractNLCTModel; nojump::Bool = false)
    if nojump
        solve_nojump(m)
    end
end


"""
```
solve_nojump(m::AbstractNLCTModel)
```
"""
function solve_nojump(m::AbstractNLCTModel)
    stategrid, functional_variables, derivatives, endogenous_variables = initialize!(m)

    if length(stategrid.x) == 1 # Univariate no jump model => use ODE methods
        s = collect(keys(get_stategrid(m)))[1] # state variable name
        ode_f, ode_callback = eqcond_nojump(m)

        tspan = (stategrid[s][1], stategrid[s][end])
        θ = parameters_to_named_tuple(map(x -> get_parameters(m)[get_keys(m)[x]], get_setting(m, :nojump_parameters)))
        prob = ODEProblem(ode_f, boundary_conditions(m)[:p][1], tspan, θ, tstops = stategrid[s][2:end - 1])
        sol = solve(prob, get_setting(m, :ode_integrator),
                    reltol = get_setting(m, :ode_reltol),
                    abstol = get_setting(m, :ode_abstol), callback = ode_callback)

        augment_variables_nojump!(m, stategrid, ode_f, functional_variables,
                          derivatives, endogenous_variables, sol)

        return stategrid, functional_variables, derivatives, endogenous_variables
    end
end
