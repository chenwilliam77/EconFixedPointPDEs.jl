using Test, OrdinaryDiffEq, HDF5, ModelConstructors, Plots
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))

regen_output = false

m = BruSan("ss3")
m[:τ] = .005
stategrid, funcvar, derivs, endo = initialize_nojump!(m)
funcvar[:vₑ] = m[:aₑ]^(-m[:γₑ]) .* stategrid[:η] .^ (1. - m[:γₑ]);
funcvar[:vₕ] = m[:aₕ]^(-m[:γₕ]) .* (1. .- stategrid[:η]) .^ (1. - m[:γₕ]);

N = length(stategrid)
θ = parameters_to_named_tuple(m.parameters)
timestep! = eqcond_nojump(m)
Δ = .8
Gs = timestep!(stategrid, funcvar, derivs, endo, θ; calculated_drift = false)
#=value_functions = (funcvar[:vₑ] .* .01, funcvar[:vₕ] .* .01)
new_value_functions = (similar(funcvar[:vₑ]), similar(funcvar[:vₕ]))
# _, _, L₁, L₂ = pseudo_transient_relaxation!(stategrid, new_value_functions, value_functions, Gs, endo[:μ_η], endo[:σ_η].^2, Δ)
new_value_functions[1] .= upwind_parabolic_pde(stategrid[:η], Gs[1], endo[:μ_η], endo[:σ_η].^2, zeros(length(stategrid)), value_functions[1], Δ)
new_value_functions[2] .= upwind_parabolic_pde(stategrid[:η], Gs[2], endo[:μ_η], endo[:σ_η].^2, zeros(length(stategrid)), value_functions[2], Δ)
funcvar[:vₑ] .= new_value_functions[1]
funcvar[:vₕ] .= new_value_functions[2]

T = 20
pnorm = 2
diff_vec = Vector{Float64}(undef, T)
diff_vec[1] = norm(vcat(map((x, y) -> x - y, value_functions, new_value_functions)), pnorm)
err_vec = Vector{Float64}(undef, T)
err_vec[1] = norm(eqn_err, pnorm)
for i in 1:length(value_functions)
    value_functions[i] .= new_value_functions[i]
end
old_value_functions = Vector{Tuple{Vector{Float64}, Vector{Float64}}}(undef, 0)
old_Δs = Vector{Float64}(undef, 0)
for t in 2:T
    @show t
    Gs, eqn_err = timestep!(stategrid, funcvar, derivs, endo, θ; calculated_drift = true)
    err_vec[t] = norm(eqn_err, pnorm)
    # pseudo_transient_relaxation!(stategrid, new_value_functions, value_functions, Gs, endo[:μ_η], endo[:σ_η].^2, Δ)
    new_value_functions[1] .= upwind_parabolic_pde(stategrid[:η], Gs[1], endo[:μ_η], endo[:σ_η].^2, zeros(length(stategrid)), value_functions[1], Δ)
    new_value_functions[2] .= upwind_parabolic_pde(stategrid[:η], Gs[2], endo[:μ_η], endo[:σ_η].^2, zeros(length(stategrid)), value_functions[2], Δ)
    funcvar[:vₑ] .= new_value_functions[1]
    funcvar[:vₕ] .= new_value_functions[2]
    push!(old_value_functions, Tuple(map(x -> copy(x), value_functions)))
    for i in 1:length(value_functions)
        value_functions[i] .= new_value_functions[i]
    end
end

for t in 2:length(old_value_functions)
    diff_vec[t] = norm(vcat(old_value_functions[t][1] - old_value_functions[t - 1][1],
                            old_value_functions[t][2] - old_value_functions[t - 1][2]), pnorm)
end
=#
