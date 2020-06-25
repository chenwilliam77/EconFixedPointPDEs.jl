"""
```
solve(m; nojump = false)
```

func_iter MUST return the values in the same order as they are in `m.functional_variables`.
Should also be clear on the requisites for func_iter.
"""
function solve(m::AbstractNLCTFPModel, init_guess::OrderedDict = OrderedDict{Symbol, Vector{Float64}}();
               nojump::Bool = false, nojump_method::Symbol = :ode, update_method::Symbol = :average,
               init_slm_coefs::OrderedDict = OrderedDict{Symbol, Vector{Float64}}(),
               individual_convergence::OrderedDict{Symbol, Vector{Float64}} = OrderedDict{Symbol, Vector{Float64}}(),
               error_calc::Symbol = :total_error, type_checks::Bool = true, return_sol::Bool = false, verbose::Symbol = :none,
               vars_for_error::Vector{Symbol} = Vector{Symbol}(undef, 0),
               has_verbose::Bool = true)

    if nojump
        return solve_nojump(m; method = nojump_method, return_sol = return_sol, verbose = verbose)
    else
        @assert !isempty(init_guess) "An initial guess for functional variables must be passed as the second argument to solve."

        ## Construct functional iteration loop
        stategrid, funcvar, derivs, endo = initialize!(m)
        func_iter = eqcond(m)
        funcvar_dict = get_functional_variables(m)
        θ = parameters_to_named_tuple(get_parameters(m))

        if type_checks
            stategrid <: StateGrid
            funcvar   <: OrderedDict
            derivs    <: OrderedDict
            endo      <: OrderedDict
        end

        # Get settings for the loop
        max_iter     = get_setting(m, :max_iter)        # maximum number of functional loops
        func_tol     = get_setting(m, :tol)             # convergence tolerance
        error_method = get_setting(m, :error_method)    # method for calculating error at the end of each loop
        func_errors = Vector{Float64}(undef, max_iter)  # track error for each iteration

        if update_method == :average
            um = (new, old) -> average_rate(new, old, get_setting(m, :learning_rate))
        elseif update_method == :pseudo_transient_continuation
            error("Updating method $update_method not implemented.")
        else
            error("Updating method $update_method not implemented.")
        end

        # We can stop updating individual functional variables prematurely.
        # The OrderedDict individual_convergence maps functional variables
        # to a two-element Vector. The first element must be either 1 or 0,
        # and the second element is the tolerance for convergence.
        converge_individually = !isempty(individual_convergence)

        # Some other set up
        proposal_funcvar = deepcopy(funcvar) # so we can calculate difference between proposal and current values
        total_time = [0.]
        guess_slm_coefs = !isempty(init_slm_coefs)

        if isempty(vars_for_error)
            vars_for_error = collect(keys(funcvar_dict))
        end

        # Start loop!
        if verbose != :none
            println("Beginning functional iteration . . .\n")
        end

        success = falses(1)
        for iter in 1:max_iter
            # Get a new guess
            if verbose != :none
                begin_time = time_ns()
            end

            if has_verbose
                if guess_slm_coefs
                    new_funcvar = converge_individually ?
                        func_iter(stategrid, funcvar, derivs, endo, θ; verbose = verbose,
                                  init_coefs = init_slm_coefs,
                                  individual_convergence = individual_convergence) :
                                      func_iter(stategrid, funcvar, derivs, endo, θ; init_coefs = init_slm_coefs, verbose = verbose)
                else
                    new_funcvar = converge_individually ? func_iter(stategrid, funcvar, derivs, endo, θ; verbose = verbose,
                                                                    individual_convergence = individual_convergence) :
                                                                        func_iter(stategrid, funcvar, derivs, endo, θ; verbose = verbose)
                end
            else
                if guess_slm_coefs
                    new_funcvar = converge_individually ?
                        func_iter(stategrid, funcvar, derivs, endo, θ; init_coefs = init_slm_coefs,
                                  individual_convergence = individual_convergence) :
                                      func_iter(stategrid, funcvar, derivs, endo, θ; init_coefs = init_slm_coefs)
                else
                    new_funcvar = converge_individually ? func_iter(stategrid, funcvar, derivs, endo, θ;
                                                                    individual_convergence = individual_convergence) :
                                                                        func_iter(stategrid, funcvar, derivs, endo, θ)
                end
            end

            # Update the guess
            for (k, v) in proposal_funcvar
                proposal_funcvar[k] = um(new_funcvar[funcvar_dict[k]], v) # um = update_method
            end

            # Calculate errors
            func_errors[iter] = calculate_func_error(new_funcvar, funcvar, error_method; vars = vars_for_error)

            if verbose != :none
                spaces1, spaces2, spaces3 = if iter < 10
                    repeat(" ", 16), repeat(" ", 14), repeat(" ", 5)
                elseif iter < 100
                    repeat(" ", 17), repeat(" ", 15), repeat(" ", 6)
                else
                    repeat(" ", 18), repeat(" ", 16), repeat(" ", 7)
                end

                println("Iteration $(iter), current error:     $(func_errors[iter]).")
                if verbose == :high
                    for k in vars_for_error
                        indiv_funcvar_err = calculate_func_error(proposal_funcvar[k], funcvar[k], error_method)
                        println("Error for $(k):               $indiv_funcvar_err")
                    end
                else
                    println("")
                end

                loop_time = (time_ns() - begin_time) / 6e10 # 60 * 1e9 = 6e10
                total_time[1] += loop_time
                expected_time_remaining = (max_iter - iter) * loop_time / 6e10
                println(verbose, :high, "Duration of loop:" * spaces1 * "$loop_time")
                println("Total elapsed time:" * spaces2 * "$(total_time[1])")
                println(verbose, :high, "Expected max remaining time:" * spaces3 * "$expected_time_remaining")
                println("\n")
            end

            # Convergence?
            if func_errors[iter] < func_tol
                if verbose != :none
                    println("Convergence achieved! Final round error: func_errors[iter]")
                end
                success[1] = true
                break
            else # If not, update the guesses for the functional variables
                for (k, v) in proposal_funcvar
                    funcvar[k] .= v
                end
            end
        end

        if success[1]
            if verbose != :none
                println("Calculating remaining variables . . .")
                aug_time = time_ns()
            end

            augment_variables!(m, stategrid, funcvar, derivs, endo)

            if verbose != :none
                total_time[1] += (time_ns() - aug_time) / 6e10
            end
        end

        if verbose != :none
            println("Total elapsed time: $(total_time[1])")
        end
    end
end

"""
```
solve_nojump(m::AbstractNLCTModel)
```

solves the no-jump equilibrium in `m`. The only available method currently is `:ode`, which uses
ODE methods. Planned extensions include `:pseudo_transient_continuation` via EconPDEs.jl
and Chebyshev/Smolyak projection via BasisMatrices.jl/SmolyakApprox.jl
"""
function solve_nojump(m::AbstractNLCTFPModel; method::Symbol = :ode, return_sol::Bool = false, verbose::Symbol = :none)
    stategrid, functional_variables, derivatives, endogenous_variables = initialize_nojump!(m)

    if (method == :ode || method == :ODE) && ndims(stategrid) == 1 # Univariate no jump model => use ODE methods
        s = collect(keys(get_state_variables(m)))[1] # state variable name
        ode_f, ode_callback = eqcond_nojump(m)

        tspan = (stategrid[s][1], stategrid[s][end])
        θ = parameters_to_named_tuple(map(x -> get_parameters(m)[get_keys(m)[x]], get_setting(m, :nojump_parameters)))
        prob = ODEProblem(ode_f, boundary_conditions(m)[:p][1], tspan, θ, tstops = stategrid[s][2:end - 1])
        sol = solve(prob, get_setting(m, :ode_integrator),
                    reltol = get_setting(m, :ode_reltol),
                    abstol = get_setting(m, :ode_abstol), callback = ode_callback)

        augment_variables_nojump!(m, stategrid, ode_f, functional_variables,
                                  derivatives, endogenous_variables, sol)

        if return_sol
            return stategrid, functional_variables, derivatives, endogenous_variables, sol
        else
            return stategrid, functional_variables, derivatives, endogenous_variables
        end
    elseif false
        # Add implementation for pseudo-transient continuation
    end
end

# TO DO: write a copy of solve_nojump or a wrapper that takes in specifically an AbstractNLCTDiffusionModel,
# which is for models that do not have any jumps
