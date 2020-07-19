using Test
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))

@testset "Instantiation" begin
    Li2020()
    @test isa(description(m), String) # Check no error with description
end

m = Li2020("ss0")

@testset "Fields of Li2020" begin
    for req_field in [:parameters, :keys, :state_variables, :functional_variables, :endogenous_variables, :exogenous_shocks,
                      :observables, :pseudo_observables, :spec, :subspec, :settings,
                      :test_settings, :rng, :testing, :observable_mappings, :pseudo_observable_mappings]
        @test hasfield(Li2020{Float64}, req_field)
    end
end

@testset "Variables, Shocks, Parameters, and Settings" begin
    @test haskey(get_state_variables(m), :w)

    for k in [:p, :Q̂, :xg]
        @test haskey(get_functional_variables(m), k)
    end

    for k in [:ψ, :xK, :yK, :yg, :σp, :σ, :σh, :σw, :μR_rd, :rd_rg, :rd_rg_H, :μb_μh, :μw, :μp, :μK, :μR, :rd, :rg, :rd_rf,
                                    :μb, :μh, :invst, :lvg, :κp, :κb, :κd, :κh, :κfs, :firesale_jump, :κw, :liq_prem, :bank_liq_frac,
                                    :δ_x, :indic, :rf, :rh, :K_growth, :κK]
        @test haskey(get_endogenous_variables(m), k)
    end

    for k in [:K_sh, :N_sh]
        @test haskey(get_exogenous_shocks(m), k)
    end

    for k in [:Φ, :∂Φ, :v₀, :damping_function, :N, :stategrid_method, :stategrid_dimensions, :stategrid_splice,
              :max_iterations, :boundary_conditions, :essentially_one, :avg_gdp, :liq_gdp_ratio,
              :dt, :ode_integrator, :ode_reltol, :ode_abstol, :tol, :learning_rate, :max_iter,
              :error_method, :v₀, :damping_function, :p₀_perturb, :κp_grid, :p_interpolant,
              :xK_interpolant, :xg_interpolant, :κp_interpolant, :Q̂_interpolant, :inside_iteration_nlsove_tol,
              :xg_tol, :yg_tol, :p_tol, :firesale_bound, :firesale_interpolant, :N_GH,
              :Q̂_tol, :Q̂_max_it, :dt, :nojump_parameters]
        @test haskey(get_settings(m), k)
    end

    param_keys = map(x -> x.key, get_parameters(m))
    for k in [:AH, :AL, :λ, :β, :α, :ϵ, :θ, :π, :δ, :ρ, :σK, :χ, :η]
        @test k in param_keys
    end
end
