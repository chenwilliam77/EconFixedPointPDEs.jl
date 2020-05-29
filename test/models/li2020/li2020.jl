using Test
include("../../../src/includeall.jl")

@testset "Instantiation" begin
    Li2020()
    @test isstring(description(m)) # Check no error with description
end

m = Li2020("ss0")

@testset "Fields of Li2020" begin
    for req_field in [:parameters, :keys, :stategrid, :endogenous_variables, :exogenous_shocks,
                      :observables, :pseudo_observables, :spec, :subspec, :settings,
                      :test_settings, :rng, :testing, :observable_mappings, :pseudo_observable_mappings]
        @test hasfield(m, req_field)
    end
end

@testset "Variables, Shocks, Parameters, and Settings" begin
    @test haskey(m.stategrid, :w)

    for k in [:p, :Q, :ψ]
        @test haskey(get_endogenous_variables(m), k)
    end

    for k in [:K_sh, :N_sh]
        @test haskey(get_exogenous_shocks(m), k)
    end

    for k in [:Φ, :∂Φ, :v₀, :vp_function, :N, :stategrid_method, :stategrid_dimensions,
              :max_iterations, :boundary_conditions, :essentially_one, :avg_gdp, :liq_gdp_ratio,
              :dt, :ode_integrator, :ode_reltol, :ode_abstol]
        @test haskey(get_settings(m), k)
    end

    param_keys = map(x -> x.key, get_parameters(m))
    for k in [:AH, :AL, :λ, :β, :α, :ϵ, :θ, :π, :δ, :ρ, :σK, :χ, :η]
        @test k in param_keys
    end
end
