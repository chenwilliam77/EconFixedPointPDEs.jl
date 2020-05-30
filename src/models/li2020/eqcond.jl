"""
```
eqcond_nojump(m::Li2020)
```

constructs the ODE characterizing equilibrium
with no jumps and the events deciding termination
"""
function eqcond_nojump(m::Li2020)
    Φ  = get_setting(m, :Φ)
    ∂Φ = get_setting(m, :∂Φ)
    ∂p∂w0, ∂p∂wN = get_setting(m, :boundary_conditions)[:∂p∂w]

    f1 = function _ode_nojump_li2020(p, θ, w)
        if w == 0.
            ∂p∂w = ∂p∂w0
        elseif w == 1.
            ∂p∂w = ∂p∂wN
        else
            ψ = max(min((Φ(p, θ[:χ], θ[:δ]) + θ[:ρ] * p - θ[:AL]) / (θ[:AH] - θ[:AL]), 1), 0)
            xK = (ψ / w)
            yK = (1 - ψ) / (1 - w)
            σp = sqrt((θ[:AH] - θ[:AL]) / (p * (xK - yK))) - θ[:σK]
            σ  = xK * (θ[:σK] + σp)
            σh = yK * (θ[:σK] + σp)

            ∂p∂w = max.(0., σp ./ (w .* (1 - w) .* (xK - yK) .* (θ[:σK] + σp)))
        end

        return ∂p∂w
    end

    ode_condition(p, w, integrator) = p - get_setting(m, :boundary_conditions)[:p][2]
    ode_affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(ode_condition, ode_affect!)

    return f1, cb
end
