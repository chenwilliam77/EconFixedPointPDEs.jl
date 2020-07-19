"""
```
eqcond_nojump(m::BruSan)
```

constructs the ODE characterizing equilibrium
with no jumps and the events deciding termination
"""
function eqcond_nojump(m::BruSan)
    f = if m[:γₑ].value == 1. && m[:γₕ].value == 1. && m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        function _ode_nojump_brusan_log(q, θ, η)
            # Set up
            @unpack ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ

            # Calculate capital-related quantities
            ι = (χ₁ * q - 1.) / χ₂
            Φ = χ₁ / χ₂ * log(χ₁ * q)

            # Solve for φ_e from market-clearing for consumption
            #=
            aₑ φ_e * η + aₕ * φ_h * (1. - η)
            = aₑ φ_e * η + aₕ * (1 - φ_e * η) / (1 - η) * (1. - η)
            = aₑ φ_e * η + aₕ * (1 - φ_e * η)
            = aₕ + (aₑ - aₕ) * φ_e * η
            q * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1 - η)) = aₕ + (aₑ - aₕ) * φₑ * η
            =#
            cₑ_per_nₑ = ρₑ + νₑ
            cₕ_per_nₕ = ρₕ + νₕ
            φ_e = ((q * (χ₂ * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1. - η)) + χ₁) - 1.) / χ₂ - aₕ) / (aₑ - aₕ) / η
            φ_h = (1. - φ_e * η) / (1. - η)

            # Calculate volatilities
            σ_q = if φ_e > φ_h
                sqrt(((aₑ - aₕ) / q ) / (φ_e - φ_h)) - σ
            elseif φ_e < φ_h
                real(sqrt(Complex(((aₑ - aₕ) / q ) / (φ_e - φ_h)))) - σ
            else
                0.
            end
            # debt_share = η * (φ_e - 1.) # Share of aggregate wealth held in debt
            σ_η  = (φ_e - 1) * (σ + σ_q)
            ησ_η = η * σ_η

            # Compute drifts
            # μ_η = (aₑ - ι) / q - cₑ_per_nₑ - τ + ((φ_e - 1.) * (σ + σ_q))^2
            μ_η  = (aₑ - ι) / q - cₑ_per_nₑ - τ + σ_η^2
            ημ_η = η * μ_η
            # μ_q  = qη / q * ημ_η + .5 * qηη / q * (ησ_η)^2

            # Compute interest rate
            # dr_f = (aₑ - ι) / q + μ_q + Φ - δ + σ * σ_q - ςₑ

            # Return ∂q_∂η = q * σ_q / ησ_η
            ∂q_∂η = q * σ_q / ησ_η

            return ∂q_∂η
        end
    end

    cb = create_callback(m)

    return f, cb
end

function create_callback(m::BruSan)
    odecond = if m[:γₑ].value == 1. && m[:γₕ].value == 1. && m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        function _ode_condition_brusan_log(q, η, integrator)
            cₑ_per_nₑ = m[:ρₑ] + m[:νₑ]
            cₕ_per_nₕ = m[:ρₕ] + m[:νₕ]
            φ_e = ((q * (m[:χ₂] * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1. - η)) + m[:χ₁]) - 1.) /
                   m[:χ₂] - m[:aₕ]) / (m[:aₑ] - m[:aₕ]) / η
            # φ_e = (q * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1 - η)) - m.aₕ) / (m.aₑ - m.aₕ) / η
            return φ_e * η - 1.
        end
    end

    ode_affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(odecond, ode_affect!)

    return cb
end

#=
mpara = Dict(:ρₑ => m.ρₑ, :ρₕ => m.ρₕ, :νₑ => m.νₑ, :νₕ => m.νₕ,
             :aₑ => m.aₑ, :aₕ => m.aₕ, :σ => m.σ, :δ => m.δ,
             :χ₁ => m.χ₁, :χ₂ => m.χ₂, :τ => m.τ)
a = [1.]
b = [1.05]
c = [0.]
for i in 1:10
    c[1] = (a[1] + b[1]) / 2
    prob = ODEProblem(odef, q₀ * c[1], (stategrid[:η][1], stategrid[:η][end]), mpara, tstops = stategrid[:η][2:end - 1])
    try
        sol = solve(prob, Tsit5(), callback = cb)
    catch e
        b[1] = c[1]
    end
    if b[1] != c[1]
        break
    end
end

prob = ODEProblem(odef, q₀ * c[1], (stategrid[:η][1], stategrid[:η][end]), mpara, tstops = stategrid[:η][2:end - 1])
sol = solve(prob, Tsit5(), callback = cb)=#
# If this fails, then try RK4! And if that fails, try Euler!
