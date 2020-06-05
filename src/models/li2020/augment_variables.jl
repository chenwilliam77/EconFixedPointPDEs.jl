"""
```
augment_variables_nojump!(m::Li2020, stategrid::StateGrid, ode_f::Function, diffvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of differential equations.
"""
function augment_variables_nojump!(m::Li2020, stategrid::StateGrid, ode_f::Function, diffvar::OrderedDict{Symbol, AbstractVector{S}},
                                   derivs::OrderedDict{Symbol, AbstractVector{S}},
                                   endo::OrderedDict{Symbol, AbstractVector{S}}, odesol::ODESolution) where {S <: Real}

    # Unpack equilibrium endogenous variables and stategrid
    p    = diffvar[:p]
    derivs[:∂p∂w] = similar(p)
    ∂p∂w = derivs[:∂p∂w]
    ψ    = endo[:ψ]
    xK   = endo[:xK]
    yK   = endo[:yK]
    xg   = endo[:xg]
    yg   = endo[:yg]
    σp   = endo[:σp]
    σ    = endo[:σ]
    σh   = endo[:σh]
    w    = stategrid[:w]

    # Interpolate remainder of the solution for q from odesol
    i = findfirst(stategrid[:w] .>= odesol.t[end])
    p[1:i] .= odesol(stategrid[:w][1:i])
    p[(i + 1):end] .= boundary_conditions(m)[:p][2]
    θ = parameters_to_named_tuple(map(x -> get_parameters(m)[get_keys(m)[x]], get_setting(m, :nojump_parameters)))
    ∂p∂w[1:i] .= map(j -> ode_f(p[j], θ, stategrid[:w][j]), 1:i)
    @show ∂p∂w[1:i]
    ∂p∂w[(i + 1):end] .= 0.

    # Solve for values not calculated during the calculation of equilibrium
    Φ = get_setting(m, :Φ)
    ψ[1:i] .= map(x -> max.(min.((Φ(x, θ[:χ], θ[:δ]) + θ[:ρ] * x - θ[:AL]) ./ (θ[:AH] - θ[:AL]), 1.), 0.), p[1:i])
    ψ[(i + 1):end] .= 1.
    xK .= (ψ ./ w)
    yK .= (1. .- ψ) ./ (1. .- w)
    xg .= zeros(eltype(w), length(w))
    yg .= fill(get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level), length(w)) ./ p ./ (1. .- w)
    ψ_is1 = (i + 1):length(w)
    ψ_no1 = 2:i
    σp[ψ_no1] .= sqrt.((θ[:AH] - θ[:AL]) ./ (p[ψ_no1] .* (xK[ψ_no1] - yK[ψ_no1]))) .- θ[:σK]
    σp[ψ_is1] .= θ[:σK] .* ∂p∂w[ψ_is1] .* w[ψ_is1] .* (1. .- w[ψ_is1]) .*
        (xK[ψ_is1] - yK[ψ_is1]) ./ (1. .- ∂p∂w[ψ_is1] .* w[ψ_is1] .* (1. .- w[ψ_is1]) .* (xK[ψ_is1] - yK[ψ_is1]))
    σ  .= xK .* (θ[:σK] .+ σp)
    σh .= yK .* (θ[:σK] .+ σp)

    if 0. in w
        ∂Φ       = get_setting(m, :∂Φ)
        w0       = findfirst(w .== 0.)
        p[w0]    = get_setting(m, :boundary_conditions)[:p][1]
        d0       = (θ[:AH] - θ[:AL]) / (∂Φ(p[w0], θ[:χ]) + θ[:ρ]) * ((θ[:AH] - θ[:AL]) / (p[w0] * θ[:σK])^2 * p[w0] + 1)
        ψ[w0]    = 0.
        ∂p∂w[w0] = d0
        xK[w0]   = 0.
        yK[w0]   = 1.
        yg[w0]   = endo[:Q][w0] / p[w0]
        σp[w0]   = 0.
        σ[w0]    = 0.
        σh[w0]   = yK[w0] * θ[:σK]
    end

    if 1. in w
        wN       = findfirst(w .== 1.)
        ∂p∂w[wN] = 0. # INTERPOLATE using solved price function somehow, maybe just linear interpolation, also maybe boundary condition
        xK[wN]   = 1.
        yK[wN]   = 0.
        xg[wN]   = 0.
        yg[wN]   = 1 / p[wN] * endo[:Q][wN - 1] / (1 - w[wN - 1])
        σp[wN]   = 0.
        σ[wN]    = xK[wN] * θ[:σK]
        σh[wN]   = 0.
    end
end
