"""
```
augment_variables!(m::Li2020, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}) where {S <: Real}

augment_variables_nojump!(m::Li2020, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}

augment_variables_nojump!(m::Li2020, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of functional equations.
"""
function augment_variables(m::Li2020, stategrid::StateGrid, funcvar::OrderedDict{Symbol, AbstractVector{S}},
                           derivs::OrderedDict{Symbol, AbstractVector{S}},
                           endo::OrderedDict{Symbol, AbstractVector{S}}) where {S <: Real}

    # Unpack equilibrium endogenous variables and stategrid
    @unpack ψ, xK, yK, yg, σp, σ, σh, σw, μR_rd, rd_rg, μb_μh, μw, μp, μK, μR, rd, rd_rf, rg, μb, μh, invst, lvg, κp, κb, κd, κh, κfs, firesale_jump, κw, liq_prem, bank_liq_frac, δ_x, indic, rf, rh, K_growth, κK = endo
    @unpack p, xg, Q̂     = funcvar
    @unpack ∂p∂w, ∂²p∂w² = derivs
    w    = stategrid[:w]
    θ    = parameters_to_named_tuple(get_parameters(m))
    f_μK = get_setting(m, :μK)
    Φ    = get_setting(m, :Φ)
    yg_tol = get_setting(m, :yg_tol)
    firesale_bound = get_setting(m, :firesale_bound)
    firesale_interpolant = get_setting(m, :firesale_interpolant)

    # Calculate various quantities
    ∂p∂w   .= differentiate(w, p)    # as Li (2020) does it
    ∂²p∂w² .= differentiate(w, ∂p∂w) # as Li (2020) does it
    invst  .= map(x -> Φ(x, θ[:χ], θ[:δ]), p)
    ψ      .= (invst .+ θ[:ρ] .* (p + Q̂) .- θ[:AL]) ./ (θ[:AH] - θ[:AL])
    ψ[ψ .> 1.] .= 1.

    # Portfolio choices
    xK   .= (ψ ./ w) .* (p ./ (p + Q̂))
    xK[1] = xK[2]
    lvg  .= xK + xg
    yK   .= (p ./ (p + Q̂) - w .* xK)  ./ (1. .- w)
    yK[1] = 1.
    yK[end] = yK[end - 1]
    Q     = get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level)
    yg   .= (Q ./ (p + Q̂) - w .* xg) ./ (1. .- w)
    yg[end] = yg[end - 1]
    yg[yg .< yg_tol] = yg_tol # avoid yg < 0
    δ_x   .= max.(θ[:β] .* (xK + xg .- 1.) - xg, 0.)
    indic .= δ_x .> 0.

    # Volatilities
    σp .= ∂p∂w .* w .* (1. .- w) .* (xK - yK) ./ (1. .- ∂p∂w .* w .* (1. .- w) .* (xK - yK)) .* θ[:σK]
    σp[end] = 0. # no volatility at the end
    σ  .= xK .* (θ[:σK] .+ σp)
    σh .= yK .* (θ[:σK] .+ σp)
    σw .= (1. .- w) .* (σ - σh)

    # Deal with post jump issues
    firesale_jump .= xK .* κp + (θ[:α] / (1 - θ[:α])) .* δ_x
    firesale_ind  = firesale_jump .< firesale_bound
    firesale_spl  = interpolate(w[firesale_ind], firesale_jump[ind], firesale_interpolant)
    extrapolate(firesale_spl, Line()) # linear extrapolation
    firesale_jump .= firesale_spl(w)

    # Jumps
    κd  .= θ[:θ] .* (xK .+ θ[:ϵ] .- 1.) ./ (xK + xg .- 1.) .* (xK .+ θ[:ϵ] .> 1.)
    κb  .= θ[:θ] .* min.(1. - θ[:ϵ], xK) + (1. .- θ[:θ]) .* firesale_jump
    κfs .= (θ[:α] / (1 - θ[:α])) .* δ_x .* w ./ (1. .- w)
    κfs[end] = 0.
    κh  .= yK .* κp + (1. .- yK .- yg) .* κd - κfs
    κw  .= 1. .- (1. .- κb) ./ (1. .- κh .- w .* (κb - κh))

    # Generate liquidity premium and price of risks
    liq_prem .= indic .* θ[:λ] .* ((1 - θ[:π]) * (1 - θ[:θ]) * θ[:α] / (1 - θ[:α])) ./ (1. .- firesale_jump)
    index    = max(argmax(li_prem) - 6., 1.)
    fit      = SLM(w[index:end], liq_prem[index:end], decreasing = true, right_value = 0., knots = 4)
    liq_prem .= vcat(liq_prem_vec[1:index - 1], eval(fit, w[index:end]))

    # Liquidity holding across states
    bank_liq_frac .= xg .* w ./ (Q ./ (p + Q̂))

    # Main drifts and interest rates
    μR_rd   .= (θ[:σK] + σp) .^ 2 .* xK - θ[:AH] ./ p + (θ[:λ] * (1 - θ[:θ])) .*
        (κp + indic .* (θ[:α] / (1 - θ[:α]) * θ[:β])) ./ (1. .- firesale_jump) +
         (xK .+ θ[:ϵ] .< 1.) .* (θ[:λ] * θ[:θ]) ./ (1. .- xK)
    rd_rg   .= (θ[:λ] * (1 - θ[:θ]) * θ[:α] / (1 - θ[:α]) * (1 - θ[:β])) .* indic ./ (1. .- firesale_jump)
    rd_rg_H .= θ[:λ] .* κd ./ (1. .- yK .* κp - (1. .- yK .- yg) .* κd + κfs)
    index_xK = xK .<= 1. # In this scenario, the rd-rg difference must be solved from hh's FOC
    rd_rg[index_xK] = rd_rg_H[index_xK]
    μb_μh   .= (xK - yK) .* μR_rd + (xK .* θ[:AH] - yK .* θ[:AL]) ./ p - (xg - yg) .* rd_rg
    μw      .= (1. .- w) .* (μb_μh + σh .^ 2 - σ .* σh - w .* (σ - σh) .^ 2 -
                       θ[:η] ./ (1. .- w))
    μw[end]  = μw[end - 1] # drift should be similar at the end
    μp      .= ∂p∂w .* w .* μw + (1. / 2.) .* ∂²p∂w² .* (w .* (1. .- w) .* (σ - σh)) .^ 2
    μp[end]  = μp[end - 1] # drift should be similar at the end
    μK      .= map(x -> f_μK(x, θ[:χ], θ[:δ]), p)
    μR      .= μp .- θ[:δ] + μK + θ[:σK] .* σp - invst ./ p
    Kgrowth .= μK .- θ[:δ]
    κK      .= θ[:θ] .* ψ

    # Other interest rates
    rd    .= μR - μR_rd
    rg    .= rd - rd_rg
    rd_rf .= (θ[:λ] * (1. - θ[:θ]) * θ[:α] / (1 - θ[:α]) * (-θ[:β])) .* indic ./ (1. .- firesale_jump)
    rf    .= rd - rd_rf
    rh    .= rd - θ[:λ] .* κd ./ (1. .- yK .* κp - (1. .- yK .- yg) .* κd .+ κfs) # risk free rate for households

    # Other growth rates
    μb .= xK .* (μR + θ[:AH] ./ p - rd) - xg .* rd_rg + rd .- θ[:ρ]
    μh .= μb - μb_μh
end

"""
```
augment_variables_nojump!(m::Li2020, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of functional equations. The first derivative
of `p` is directly calculated using `ode_f`
"""
function augment_variables_nojump!(m::Li2020, stategrid::StateGrid, ode_f::Function, funcvar::OrderedDict{Symbol, Vector{S}},
                                   derivs::OrderedDict{Symbol, Vector{S}},
                                   endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}

    # Unpack equilibrium endogenous variables and stategrid
    p    = funcvar[:p]
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

"""
```
augment_variables_nojump_fd!(m::Li2020, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}
```

calculates derivatives and additional endogenous variables that can be computed
after solving the equilibrium system of functional equations and using
finite differences to approximate the derivatives.
"""
function augment_variables_nojump!(m::Li2020, stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                      derivs::OrderedDict{Symbol, Vector{S}},
                                      endo::OrderedDict{Symbol, Vector{S}}, odesol::ODESolution) where {S <: Real}

    # Unpack equilibrium endogenous variables and stategrid
    p    = funcvar[:p]
    derivs[:∂p_∂w] = similar(p)
    ∂p∂w   = derivs[:∂p_∂w]
    ∂²p∂w² = derivs[:∂²p_∂w²]
    @unpack ψ, xK, yK, yg, σp, σ, σh, μR_rd, rd_rg, μb_μh, μw, μp, μK, μR, rd, rg, μb, μh = endo
    w      = stategrid[:w]
    θ = parameters_to_named_tuple(map(x -> get_parameters(m)[get_keys(m)[x]], get_setting(m, :nojump_parameters)))
    f_μK = get_setting(m, :μK)

    # Interpolate remainder of the solution for q from odesol
    i = findfirst(w .>= odesol.t[end])
    p[1:i] .= odesol(w[1:i])
    p[(i + 1):end] .= boundary_conditions(m)[:p][2]
    ∂p∂w   .= differentiate(w, p)    # as Li (2020) does it
    ∂²p∂w² .= differentiate(w, ∂p∂w) # as Li (2020) does it

    # Solve for values not calculated during the calculation of equilibrium
    Φ = get_setting(m, :Φ)
    ψ[1:i] .= map(x -> max.(min.((Φ(x, θ[:χ], θ[:δ]) + θ[:ρ] * x - θ[:AL]) ./ (θ[:AH] - θ[:AL]), 1.), 0.), p[1:i])
    ψ[(i + 1):end] .= 1.
    xK .= (ψ ./ w)
    yK .= (1. .- ψ) ./ (1. .- w)
    xg .= zeros(eltype(w), length(w))
    yg .= fill(get_setting(m, :avg_gdp) * get_setting(m, :gov_bond_gdp_level), length(w)) ./ p ./ (1. .- w)
    yg[end] = yg[end - 1]
    σp .= ∂p∂w .* w .* (1. .- w) .* (xK - yK) ./ (1. .- ∂w .* w .* (1. .- w) .* (xK - yK)) .* θ[:σK]
    σp[end] = 0. # no volatility at the end
    σ  .= xK .* (θ[:σK] .+ σp)
    σh .= yK .* (θ[:σK] .+ σp)

    # Calculate remaining objects
    invst   = map(x -> Φ(x, θ[:χ], θ[:δ]), p)
    μR_rd  .= (θ[:σK] + σp) .^ 2 .* xK - θ[:AH] ./ p
    rd_rg  .= 0.
    μb_μh  .= (xK - yK) .* μR_rd + (xK .* θ[:AH] - yK .* θ[:AL]) ./ p - (xg - yg) .* rd_rg
    μw     .= (1. .- w) .* (μb_μh + σh .^ 2 - σ .* σh - w .* (σ - σh) .^ 2 -
                       θ[:η] ./ (1. .- w))
    μw[end] = μw[end - 1] # drift should be similar at the end
    μp     .= ∂p∂w .* w .* μw + (1. / 2.) .* ∂²∂w² .* (w .* (1. .- w) .* (σ - σh)) .^ 2
    μp[end] = μp[end - 1] # drift should be similar at the end
    μK     .= map(x -> f_μK(x, θ[:χ], θ[:δ]), p)
    μR     .= μp .- θ[:δ] + μK + θ[:σK] .* σp - invst ./ p
    rd     .= μR - μR_rd
    rg      = copy(rd)
    μb     .= xK .* (μR + θ[:AH] ./ p - rd) - xg .* rd_rg + rd .- θ[:ρ]
    μh     .= μb - μb_μh
end
