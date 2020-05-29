# This function augments the states carried around by Li2020 with additional quantities that can
# be computed from the solved equilibrium.

"""
```
augment_endogenous_variables_nojump(m::Li2020)
```

calculates additional endogenous variables that can be computed
after solving the equilibrium system of differential equations.
"""
function augment_endogenous_variables_nojump(m::Li2020, grid::NamedTuple, endo::OrderedDict{Symbol, Vector{S}}) where {S <: Real}
    # Unpack equilibrium endogenous variables and stategrid
    p    = endo[:p]
    ∂p∂w = endo[:∂p∂w]
    ψ    = endo[:ψ]
    xK   = endo[:xK]
    yK   = endo[:yK]
    xg   = endo[:xg]
    yg   = endo[:yg]
    σp   = endo[:σp]
    σ    = endo[:σ]
    σh   = endo[:σh]
    w    = grid[:w]

    # Solve for values not calculated during the calculation of equilibrium
    endo[:ψ]  .= max.(min.((Φ(p) + m[:ρ] .* p - m[:AL]) ./ (m[:AH] - m[:AL]), 1), 0.)
    ψ_is1 = findall(ψ .>= get_setting(m, :essentially_one))
    ψ_no1 = .!(ψ_is1)
    endo[:xK] .= (ψ ./ w)
    endo[:yK] .= (1. .- ψ) ./ (1. .- w)
    endo[:xg] .= zeros(eltype(w), length(w))
    endo[:yg] .= fill(get_setting(m, :avg_gdp) * get_setting(m, :liq_gdp_ratio), length(w)) ./ p ./ (1. .- w)
    endo[:σp][ψ_no1] = sqrt((m[:AH] - m[:AL]) ./ (p .* (xK - yK))) - m[:σK]
    endo[:σp][ψ_is1] = m[:σK] .* ∂p∂w[ψ_is1] .* w[ψ_is1] .* (1. .- w[ψ_is1]) .*
        (xK[ψ_is1] - yK[ψ_is1]) ./ (1. .- ∂p∂w[ψ_is1] .* w[ψ_is1] .* (1. .- w[ψ_is1]) .* (xK[ψ_is1] - yK[ψ_is1]))
    endo[:σ]  .= xK .* (m[:σK] .+ σp)
    endo[:σh] .= yK .* (m[:σK] .+ σp)

    if 0. in w
        w0       = findfirst(w .== 0.)
        p[w0]    = get_setting(m, :boundary_conditions)[:p][1]
        ψ[w0]    = (dΦ(p[w0]) + m[:ρ]) / (m[:AH] - m[:AL]) * get_setting(m, :d0)
        ∂p∂w[w0] = get_setting(m, :d0)
        xK[w0]   = 0.
        yK[w0]   = 0.
        yg[w0]   = endo[:Q][w0] / p[w0]
        σp[w0]   = 0.
        σh[w0]   = yK[w0] * m[:σK]
        σ[w0]    = xK[w0] * m[:σK]
    end

    if 1. in w
        wN       = findfirst(w .== 1.)
        ∂p∂w[wN] = 0. # INTERPOLATE using solved price function somehow, maybe just linear interpolation, also maybe boundary condition
        xK[wN]   = 1.
        yK[wN]   = 0.
        xg[wN]   = 0.
        yg[wN]   = 1 / p[wN] * endo[:Q][wN - 1] / (1 - w[wN - 1])
    end
end
