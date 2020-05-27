"""
```
eqcond_nojump(m::Li2020)
```
constructs the ODE characterizing equilibrium
with no jumps.
"""
function eqcond_nojump(m::Li2020)
    Φ  = get_setting(m, :Φ)
    ∂Φ = get_setting(m, :∂Φ)
    # pass in params as a labeled array
    f1 = function _ode_nojump_li2020(dp, p, θ, w)
        ψ = max.(min((Φ(p) + θ[:ρ] * p - θ[:AL]) / (θ[:AH] - θ[:AL]), 1), 0)
        xK = (ψ / w)
        yK = (1 - ψ) / (1 - w)
        σp = sqrt((θ[:AH] - θ[:AL]) / (p * (xK - yK))) - θ[:σK]
        σ  = xK * (θ[:σK] + σp)
        σh = yK * (θ[:σK] + σp)

        if w == 0.
            ∂p∂w = θ[:∂p∂w0]
        elseif w == 1.
            ∂p∂w = θ[:∂p∂wN]
        else
            ∂p∂w = max.(0., σp ./ (w .* (1 - w) .* (xK - yK) .* (θ[:σK] + σp)))
        end
    end

    return f1
end
