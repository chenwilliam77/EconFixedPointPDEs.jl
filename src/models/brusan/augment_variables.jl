"""
```
augment_variables!(m::BruSan, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}) where {S <: Real}

augment_variables_nojump!(m::BruSan, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of functional equations.
"""
function augment_variables!(m::BruSan, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                            derivs::OrderedDict{Symbol, Vector{S}},
                            endo::OrderedDict{Symbol, Vector{S}}) where {S <: Real}
end

"""
```
augment_variables_nojump!(m::BruSan, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of functional equations. The first derivative
of `p` is directly calculated using `ode_f`
"""
function augment_variables_nojump!(m::BruSan, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
                                   derivs::OrderedDict{Symbol, Vector{S}},
                                   endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}

    if m[:γₑ].value == 1. && m[:γₕ].value == 1. && m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        return augment_variables_nojump_log!(m, stategrid, ode_f, funcvar, derivs, endo, odesol)
    end
end


function augment_variables_nojump_log!(m::BruSan, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
                                       derivs::OrderedDict{Symbol, Vector{S}},
                                       endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}

    # Unpack equilibrium endogenous variables and stategrid
    q       = funcvar[:q]
    ∂q_∂η   = derivs[:∂q_∂η]
    ∂²q_∂η² = derivs[:∂²q_∂η²]
    η       = stategrid[:η]
    @unpack φ_e, φ_h, σ_q, ςₑ, ςₕ, μ_η, σ_η, ι, Φ, μ_K = endo
    θ = parameters_to_named_tuple(m.parameters)
    @unpack ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ
    cₑ_per_nₑ = ρₑ + νₑ
    cₕ_per_nₕ = ρₕ + νₕ

    # Interpolate remainder of the solution for q from odesol
    i = findfirst(η .>= odesol.t[end])
    q[1:i] .= odesol(η[1:i])
    ∂q_∂η[1:i] .= map(j -> ode_f(q[j], θ, η[j]), 1:i)
    if cₑ_per_nₑ == cₕ_per_nₕ
        q[(i + 1):end] .= boundary_conditions(m)[:q][2]
        ∂q_∂η[(i + 1):end] .= 0.
    else
        q[(i + 1):end] .= (χ₂ * aₑ + 1.) ./ (χ₂ .* (η[(i + 1):end] .* cₑ_per_nₑ +
                                                    (1. .- η[(i + 1):end]) .* cₕ_per_nₕ) .+ χ₁)
        ∂q_∂η[(i + 1):end] .= -(χ₂ * aₑ + 1) / (χ₂ * (η  * ce_per_n_e + (1 - η) * ch_per_nh) + χ₁)^2 *
            (χ₂ * (cₑ_per_nₑ - cₕ_per_nₕ))
    end
    ∂²q_∂η² .= CenteredDifference(2, 2, diff(η), length(η)) * RobinBC((0., 1., 0.), (0., 1., 0.)) .* q

    # Solve for values not calculated during the calculation of equilibrium
    φ_e[1:i] .= ((q[1:i] .* (χ₂ .* (cₑ_per_nₑ .* η[1:i] + cₕ_per_nₕ .* (1. .- η[1:i])) .+ χ₁) .- 1.) ./ χ₂ .- aₕ) ./
        (aₑ - aₕ) ./ η[1:i]
    φ_e[(i + 1):end] .= 1. ./ η[(i + 1):end]
    φ_h .= (1. .- φ_e .* η) ./ (1. .- η)
    is1 = (i + 1):length(η) # experts hold all capital
    no1 = 1:i
    σ_q[no1] .= sqrt.((aₑ - aₕ) ./ q ./ (φ_e[no1] - φ_h[no1])) .- σ
    σ_q[is1] .= ∂q_∂η[is1] .* η[is1] .* (φ_e[is1] .- 1.) .* σ ./
        (q[is1] .- ∂q_∂η[is1] .* η .* (φ_e[is1] .- 1.))
    ςₑ .= φ_e .* (σ .+ σ_q).^2
    ςₕ .= φ_h .* (σ .+ σ_q).^2

    ι   .= (χ₁ .* q .- 1.) ./ χ₂
    Φ   .= (χ₁ / χ₂) .* log.(χ₁ .* q)
    μ_K .= Φ .- δ

    σ_η .= (φ_e .- 1.) .* (σ .+ σ_q)
    μ_η .= (aₑ .- ι) ./ q .- cₑ_per_nₑ .- τ .+ σ_η.^2
    μ_q .= ∂q_∂η .* μ_η .* η + ∂²q_∂η² ./ 2 .* (σ_η .* η).^2
end
