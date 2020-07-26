"""
```
eqcond_nojump(m::BruSan)
```


constructs the ODE characterizing equilibrium
with no jumps and the events deciding termination
"""
function eqcond_nojump(m::BruSan)
    if m[:γₑ].value == 1. && m[:γₕ].value == 1. && m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        f = function _ode_nojump_brusan_log(q, θ, η)
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

        cb = create_callback(m)

        return f, cb

    elseif m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        # Set up difference operators
        dx = vcat(stategrid[:η][1], diff(stategrid[:η]), 1. - stategrid[:η][end])
        Q  = RobinBC((0., 1., 0.), (0., 1., 0.), dx)
        L₂ = CenteredDifference(2, 2, dx, length(stategrid)) * Q

        # Grab some settings
        q₀                 = get_setting(m, :boundary_conditions)[:q][1]
        qₙ                 = get_setting(m, :boundary_conditions)[:q][end]
        vf_interpolant     = get_setting(m, :interpolant)
        max_q₀_perturb     = get_setting(m, :max_q₀_perturb)
        ode_integrator     = get_setting(m, :ode_integrator)
        backup_integrators = get_setting(m, :backup_ode_integrators)
        ode_abstol         = get_setting(m, :ode_abstol)
        ode_reltol         = get_setting(m, :ode_reltol)
        N                  = get_setting(m, :N)

        # Form the callback for the static step
        cb = create_callback(m)

        # Construct ODE function
        function _staticstep_nojump_brusan_nonunit_gamma(q, θ::NamedTuple, η, vₑ, vₕ, ∂vₑ_∂η, ∂vₕ_∂η)
            @unpack ρₑ, ρₕ, νₑ, νₕ, γₑ, γₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ

            # Solve for φ_e from market-clearing for consumption
            cₑ_per_nₑ = ρₑ + νₑ
            cₕ_per_nₕ = ρₕ + νₕ
            φ_e = ((q * (χ₂ * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1. - η)) + χ₁) - 1.) / χ₂ - aₕ) / (aₑ - aₕ) / η
            φ_h = (1. - φ_e * η) / (1. - η)

            # Calculate σ_q
            σ_vₑ_div_ssq = ∂vₑ_∂η(η) / vₑ(η) * η * (φ_e - 1.) # σ_vₑ / (σ + σ_q)
            σ_vₕ_div_ssq = ∂vₕ_∂η(η) / vₕ(η) * η * (φ_e - 1.)
            rp_div_ssq² = (γₑ * φ_e - γₕ * φ_h) +
                ((γₑ - 1.) * σ_vₑ_div_ssq - (γₕ - 1.) * σ_vₕ_div_ssq)  # (risk premium) / (σ + σ_q)²
            σ_q = try
                sqrt(((aₑ - aₕ) / q ) / rp_div_ssq²) - σ
            catch e
                if isa(e, DomainError)
                    real(sqrt(Complex(((aₑ - aₕ) / q ) / rp_div_ssq²))) - σ
                else
                    rethrow(e)
                end
            end

            # σ_η  = (φ_e - 1) * (σ + σ_q)
            # ησ_η = η * σ_η
            # ∂q_∂η = q * σ_q / ησ_η
            ∂q_∂η = q * σ_q / (η * (φ_e - 1.) * (σ + σ_q))
            return ∂q_∂η
        end

        f2 = function _resize_nojump_brusan_nonunit_gammas!(y::Vector{S}, funcvar::OrderedDict{Symbol, Vector{S}}) where {S <: Real}
            funcvar[:vₑ] .= y[1:N]
            funcvar[:vₕ] .= y[(N + 1):end]
        end

        f1 = function _ptr_nojump_brusan_nonunit_gammas(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                                        derivs::OrderedDict{Symbol, Vector{S}},
                                                        endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple;
                                                        calculated_drift::Bool = false, max_q_iter::Int = 20,
                                                        max_q₀_perturb::Float64 = max_q₀_perturb,
                                                        verbose::Symbol = :none) where {S <: Real}
            # Set up
            @unpack ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ, γₑ, γₕ = θ
            @unpack vₑ, vₕ, q = funcvar
            @unpack φ_e, φ_h, ςₑ, ςₕ, σ_q, μ_q, σ_η, μ_η, σ_vₑ, σ_vₕ, μ_vₑ, μ_vₕ, ι, Φ, μ_K, dr_f = endo
            η = stategrid[:η]

            # Differentiate vₑ, vₕ and interpolate
            μ_η0 = calculated_drift ? endo[:μ_η] : Vector{S}(undef, 0)
            differentiate!(stategrid, funcvar, derivs, μ_η0; L₂ = L₂)
            @unpack ∂vₑ_∂η, ∂vₕ_∂η, ∂²vₑ_∂η², ∂²vₕ_∂η², ∂q_∂η, ∂²q_∂η² = derivs
            vₑ_interp     = interpolate((η, ), vₑ,     vf_interpolant)
            vₕ_interp     = interpolate((η, ), vₕ,     vf_interpolant)
            ∂vₑ_∂η_interp = interpolate((η, ), ∂vₑ_∂η, vf_interpolant)
            ∂vₕ_∂η_interp = interpolate((η, ), ∂vₕ_∂η, vf_interpolant)

            # Solve for q
            ode_f = (q, θ, η) -> _staticstep_nojump_brusan_nonunit_gamma(q, θ, η,
                                                                         vₑ_interp, vₕ_interp, ∂vₑ_∂η_interp, ∂vₕ_∂η_interp)
            tstops = η[2:end - 1]
            tspan  = (η[1], η[end])
            b = Vector{Float64}(undef, 1)
            c = Vector{Float64}(undef, 1)
            odesol = Vector{ODESolution}(undef, 0)
            for ode_integ in vcat(ode_integrator, backup_integrators)
                b[1] = max_q₀_perturb

                for i in 1:max_q_iter
                    c[1] = (1. + b[1]) / 2.
                    prob = ODEProblem(ode_f, q₀ * c[1], tspan, θ, tstops = tstops)
                    try
                        sol = solve(prob, ode_integ; reltol = ode_reltol,
                                    abstol = ode_abstol, callback = cb)
                    catch e
                        max_q₀_perturb[1] = c[1]
                    end
                    if max_q₀_perturb[1] != c[1]
                        break
                    end
                end

                if max_q₀_perturb[1] != c[1]
                    prob = ODEProblem(ode_f, q₀ * c[1], tspan, θ, tstops = tstops)
                    push!(odesol, solve(prob, ode_integ; reltol = ode_reltol,
                                        abstol = ode_abstol, callback = cb))
                    break
                end
            end
            if isempty(odesol)
                error("Failure to solve the static step.")
            end

            # Differentiate vₑ, vₕ w/new drift of η, and calculate second derivative of q
            _postprocessing_staticstep_brusan_nonunit_gammas!(stategrid, funcvar, derivs, endo, θ, odesol[1], ode_f)
            differentiate!(stategrid, funcvar, derivs, μ_η; L₂ = L₂, skipvar = Symbol[:q])
            ∂²q_∂η² .= L₂ * q

            # Prepare for HJB calculation
            cₑ_per_nₑ = ρₑ + νₑ
            cₕ_per_nₕ = ρₕ + νₕ
            μ_q  .= ∂q_∂η .* μ_η .* η + ∂²q_∂η² ./ 2 .* (σ_η .* η).^2
            no1 = findlast(φ_e .* η .< 1.)
            μ_q[no1 + 1] = μ_q[no1 + 2]
            μ_q[no1] = μ_q[no1 + 2]
            ι    .= (χ₁ .* q .- 1.) ./ χ₂
            Φ    .= (χ₁ / χ₂) .* log.(χ₁ .* q)
            μ_K  .= Φ .- δ
            σ_vₑ .= ∂vₑ_∂η ./ vₑ .* η .* σ_η
            σ_vₕ .= ∂vₕ_∂η ./ vₕ .* η .* σ_η
            μ_vₑ .= ∂vₑ_∂η ./ vₑ .* η .* μ_η + ∂²vₑ_∂η² ./ vₑ .* (η .* σ_η).^2
            μ_vₕ .= ∂vₕ_∂η ./ vₕ .* η .* μ_η + ∂²vₕ_∂η² ./ vₕ .* (η .* σ_η).^2
            dr_f .= (aₑ .- ι) ./ q + μ_K + μ_q + σ .* σ_q - ςₑ
            G_vₑ  = -((ρₑ + νₑ) .* log.(cₑ_per_nₑ ./ vₑ) .- cₑ_per_nₑ + dr_f +
                φ_e .* ςₑ - (γₑ / 2.) * (σ_vₑ.^2 + (φ_e .* (σ .+ σ_q)).^2) +
                (1 - γₑ) .* σ_vₑ .* φ_e .* (σ .+ σ_q))
            G_vₕ  = -((ρₕ + νₕ) .* log.(cₕ_per_nₕ ./ vₕ) .- cₕ_per_nₕ + dr_f +
                φ_h .* ςₕ - (γₕ / 2.) * (σ_vₕ.^2 + (φ_h .* (σ .+ σ_q)).^2) +
                (1 - γₕ) .* σ_vₕ .* φ_h .* (σ .+ σ_q))
            err_vₑ = (μ_vₑ - G_vₑ) .* vₑ
            err_vₕ = (μ_vₕ - G_vₕ) .* vₕ
            # neg_∂vₑ_∂t = vₑ .* ((ρₑ + νₑ) .* log.(cₑ_per_nₑ ./ vₑ) .- cₑ_per_nₑ + μ_vₑ + dr_f +
            #     φ_e .* ςₑ - (γₑ / 2.) * (σ_vₑ.^2 + (φ_e .* (σ .+ σ_q)).^2) +
            #     (1 - γₑ) .* σ_vₑ .* φ_e .* (σ .+ σ_q))
            # neg_∂vₕ_∂t = vₕ .* ((ρₕ + νₕ) .* log.(cₕ_per_nₕ ./ vₕ) .- cₕ_per_nₕ + μ_vₕ + dr_f +
            #     φ_h .* ςₕ - (γₕ / 2.) * (σ_vₕ.^2 + (φ_h .* (σ .+ σ_q)).^2) +
            #     (1 - γₕ) .* σ_vₕ .* φ_h .* (σ .+ σ_q))

            # return vcat(neg_∂vₑ_∂t, neg_∂vₕ_∂t)
            return (G_vₑ, G_vₕ), vcat(err_vₑ, err_vₕ)
        end

        return f1, f2
    elseif m[:ψₑ].value == 1. / m[:γₑ].value && m[:ψₕ].value == 1. / m[:γₕ].value
        # Set up difference operators
        dx = nonuniform_ghost_node_grid(stategrid)
        Q  = RobinBC((0., 1., 0.), (0., 1., 0.), dx)
        L₂ = CenteredDifference(2, 2, dx, length(stategrid)) * Q

        # Grab some settings
        # q₀                 = get_setting(m, :boundary_conditions)[:q][1]
        # qₙ                 = get_setting(m, :boundary_conditions)[:q][end]
        vf_interpolant     = get_setting(m, :interpolant)
        max_q₀_perturb     = get_setting(m, :max_q₀_perturb)
        ode_integrator     = get_setting(m, :ode_integrator)
        backup_integrators = get_setting(m, :backup_ode_integrators)
        ode_abstol         = get_setting(m, :ode_abstol)
        ode_reltol         = get_setting(m, :ode_reltol)
        N                  = get_setting(m, :N)

        # Form the callback for the static step
        cb = create_callback(m, interpolate((stategrid[:η], ), funcvar[:vₑ], vf_interpolant),
                             interpolate((stategrid[:η], ), funcvar[:vₕ], vf_interpolant))

        # Construct static step
        function _staticstep_nojump_brusan_crra!(F, x, θ::NamedTuple, η, dη, vₑ, vₕ, ∂vₑ_∂η, ∂vₕ_∂η, qold)
            @unpack ρₑ, ρₕ, νₑ, νₕ, γₑ, γₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ
            q   = x[1]
            ssq = x[2]
            φ_e = x[3] / η

            # Solve for φ_e from market-clearing for consumption
            ι            = (χ₁ * q - 1.) / χ₂
            σ_vₑ_div_ssq = ∂vₑ_∂η / vₑ * η * (φ_e - 1.) # σ_vₑ / (σ + σ_q)
            σ_vₕ_div_ssq = ∂vₕ_∂η / vₕ * η * (φ_e - 1.)

            F[1] = (log(q) + log(η / vₑ)) / γₑ + (log(q) + log((1. - η) / vₕ)) / γₕ - log(aₕ + (aₑ - aₕ) * φ_e * η - ι)
            F[2] = ssq * (q - (q - qold) * η * (φ_e - 1) / dη) - σ * q
            F[3] = aₑ - aₕ - q * (σ_vₕ_div_ssq - σ_vₑ_div_ssq + (φ_e - 1) / (1 - η)) * ssq^2
        end

#=        function _staticstep_nojump_brusan_crra(q, θ::NamedTuple, η, vₑ, vₕ, ∂vₑ_∂η, ∂vₕ_∂η)
            @unpack ρₑ, ρₕ, νₑ, νₕ, γₑ, γₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ

            # Solve for φ_e from market-clearing for consumption
            cₑ_per_K = (η * q / vₑ(η))^(1 / m[:γₑ])
            cₕ_per_K = ((1 - η) * q / vₕ(η))^(1 / m[:γₕ])
            ι        = (m[:χ₁] * q - 1.) / m[:χ₂]
            φ_e      = ((cₑ_per_K + cₕ_per_K) + ι - m[:aₕ]) / (m[:aₑ] - m[:aₕ]) / η
            φ_h      = (1. - φ_e * η) / (1. - η)

            # Calculate σ_q
            σ_vₑ_div_ssq = ∂vₑ_∂η(η) / vₑ(η) * η * (φ_e - 1.) # σ_vₑ / (σ + σ_q)
            σ_vₕ_div_ssq = ∂vₕ_∂η(η) / vₕ(η) * η * (φ_e - 1.)
            rp_div_ssq²  = σ_vₕ_div_ssq - σ_vₑ_div_ssq + (φ_e - 1) / (1 - η)
            σ_q = try
                sqrt(((aₑ - aₕ) / q ) / rp_div_ssq²) - σ
            catch e
                if isa(e, DomainError)
                    real(sqrt(Complex(((aₑ - aₕ) / q ) / rp_div_ssq²))) - σ
                else
                    rethrow(e)
                end
            end

            # σ_η  = (φ_e - 1) * (σ + σ_q)
            # ησ_η = η * σ_η
            # ∂q_∂η = q * σ_q / ησ_η
            ∂q_∂η = q * σ_q / (η * (φ_e - 1.) * (σ + σ_q))
            return ∂q_∂η
        end
=#

        f = function _ptr_nojump_brusan_crra(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                              derivs::OrderedDict{Symbol, Vector{S}},
                                              endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple;
                                              calculated_drift::Bool = false, max_q_iter::Int = 20,
                                              max_q₀_perturb::Float64 = max_q₀_perturb,
                                              verbose::Symbol = :none) where {S <: Real}
            # Set up
            @unpack ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ, γₑ, γₕ = θ
            @unpack vₑ, vₕ, q = funcvar
            @unpack φ_e, φ_h, ςₑ, ςₕ, σ_q, μ_q, σ_η, μ_η, σ_vₑ, σ_vₕ, μ_vₑ, μ_vₕ, ι, Φ, μ_K, dr_f = endo
            η = stategrid[:η]

            # Differentiate vₑ, vₕ and interpolate
            μ_η0 = calculated_drift ? endo[:μ_η] : Vector{S}(undef, 0)
            differentiate!(stategrid, funcvar, derivs, μ_η0; L₂ = L₂, skipvar = [:q])
            @unpack ∂vₑ_∂η, ∂vₕ_∂η, ∂²vₑ_∂η², ∂²vₕ_∂η², ∂q_∂η, ∂²q_∂η² = derivs
#=            vₑ_interp     = interpolate((η, ), vₑ,     vf_interpolant)
            vₕ_interp     = interpolate((η, ), vₕ,     vf_interpolant)
            ∂vₑ_∂η_interp = interpolate((η, ), ∂vₑ_∂η, vf_interpolant)
            ∂vₕ_∂η_interp = interpolate((η, ), ∂vₕ_∂η, vf_interpolant)=#

            # Solve for q
            #=ode_f = (q, θ, η) -> _staticstep_nojump_brusan_crra(q, θ, η,
                                                                vₑ_interp, vₕ_interp, ∂vₑ_∂η_interp, ∂vₕ_∂η_interp)
            tstops = η[2:end - 1]
            tspan  = (η[1], η[end])=#

            # Calculate initial condiiton
            qL = Float64[0]
            qR = Float64[aₑ * χ₂ + 1.]
            q₀ = Float64[0.]
            GS1 = (η[1] / vₑ[1])^(1. / γₑ)
            GS2 = ((1. - η[1]) / vₕ[1])^(1. / γₕ)
            GS  =  GS1 + GS2;
            for k in 1:30
                q₀[1] = (qL[1] + qR[1]) / 2.
                ι₀    = (χ₁ * q₀[1] - 1.) / χ₂
                A₀    = η[1] * (aₑ - aₕ) + aₕ - ι₀
                if log(q₀[1]) / γₕ + log(GS) > log(A₀)
                    qR[1] = q₀[1]
                else
                    qL[1] = q₀[1]
                end
            end

            #=prob = ODEProblem(ode_f, q₀[1], tspan, θ, tstops = tstops)
            odesol = solve(prob, RK4(), dt = η[2] - η[1]; reltol = ode_reltol,
                           abstol = ode_abstol, callback = cb)

            # Differentiate vₑ, vₕ w/new drift of η, and calculate second derivative of q
            _postprocessing_staticstep_brusan_crra!(stategrid, funcvar, derivs, endo, θ, odesol, ode_f)=#
            q[1]   = q₀[1]
            σ_q[1] = 0.
            φ_e[1] = 0.
            for i in 2:N
#=        QN(1,:) = [1/(q*gamma) + 1/((a - a_)*psi        + a_ - (q - 1)/kappa)/kappa,  ...
                                -(a - a_)/((a - a_)*psi + a_ - (q - 1)/kappa), 0];

        QN(2,:) = [ssq*(1 - (chi_*psi - Eta(n))/dX(n-1)) - sigma,  ...
                    -ssq*(q - q_old)*chi_/dX(n-1),  q - (q - q_old)*(chi_*psi - Eta(n))/dX(n-1)];

        QN(3,:) = [- chi_*(chi_*psi - Eta(n))*ssq^2*VVlp(n-1), ...
                    -q*chi_^2*ssq^2*VVlp(n-1), -2*q*chi_*(chi_*psi - Eta(n))*ssq*VVlp(n-1)];=#

#=                ER = zeros(3)
                QN = zeros(3, 3)
                xguess = [q[i - 1], σ + σ_q[i - 1], φ_e[i - 1] * η[1]]
                _staticstep_nojump_brusan_crra!(ER, xguess,
                                                θ, η[i], η[i] - η[i - 1], vₑ[i], vₕ[i], ∂vₑ_∂η[i], ∂vₕ_∂η[i], q[i - 1])
                QN[1,:] = [1/(xguess[1]*gamma) + 1/((a - a_)*xguess[2]        + a_ - (xguess[1] - 1)/kappa)/kappa,  ...
                           -(a - a_)/((a - a_)*xguess[3] + a_ - (xguess[1] - 1)/kappa), 0];

                QN[2,:] = [xguess[2]*(1 - (xguess[3] - Eta(n))/dX(n-1)) - sigma,  ...
                           -xguess[2]*(q - q_old)/dX(n-1),  xguess[1] - (xguess[1] - qold)*(xguess[3] - Eta(n))/dX(n-1)];

                QN[3,:] = [- (xguess[3] - Eta(n))*xguess[2]^2*VVlp(n-1), ...
                           -xguess[1] * xguess[2]^2 * VVlp(n-1), -2 * xguess[1] * (xguess[3] - Eta(n)) * xguess[2] * VVlp(n-1)];=#
                @show xguess
                @show QN
                @show ER
                EN = xguess - QN \ ER
                #=out = nlsolve((F, x) -> _staticstep_nojump_brusan_crra!(F, x, θ, η[i], η[i] - η[i - 1],
                                                                        vₑ[i], vₕ[i], ∂vₑ_∂η[i], ∂vₕ_∂η[i], q[i - 1]),
                              method = :trust_region, [q[i - 1], σ + σ_q[i - 1], φ_e[i - 1]])=#
                @show (i, η[i], EN)
                if EN[3] * η >= 1.
                    break
                else
                    q[i]   = EN[1]
                    σ_q[i] = EN[2] - σ
                    φ_e[i] = EN[3] / η
                end
            end
            @assert false
            _postprocessing_staticstep_brusan_crra!(stategrid, funcvar, derivs, endo, θ)
            differentiate!(stategrid, funcvar, derivs, μ_η; L₂ = L₂, skipvar = Symbol[:q])

            # Prepare for HJB calculation
            cₑ_per_nₑ = ρₑ + νₑ
            cₕ_per_nₕ = ρₕ + νₕ
            μ_q  .= ∂q_∂η .* μ_η .* η + ∂²q_∂η² ./ 2 .* (σ_η .* η).^2
            no1 = findlast(φ_e .* η .< 1.)
            μ_q[no1 + 1] = μ_q[no1 + 2]
            μ_q[no1] = μ_q[no1 + 2]
            ι    .= (χ₁ .* q .- 1.) ./ χ₂
            Φ    .= (χ₁ / χ₂) .* log.(χ₁ .* q)
            μ_K  .= Φ .- δ
            σ_vₑ .= ∂vₑ_∂η ./ vₑ .* η .* σ_η
            σ_vₕ .= ∂vₕ_∂η ./ vₕ .* η .* σ_η
            μ_vₑ .= ∂vₑ_∂η ./ vₑ .* η .* μ_η + ∂²vₑ_∂η² ./ vₑ .* (η .* σ_η).^2
            μ_vₕ .= ∂vₕ_∂η ./ vₕ .* η .* μ_η + ∂²vₕ_∂η² ./ vₕ .* (η .* σ_η).^2
            dr_f .= (aₑ .- ι) ./ q + μ_K + μ_q + σ .* σ_q - ςₑ
            G_vₑ  = (ρₑ + νₑ) - (η .* q).^(1. / γₑ - 1.) ./ (vₑ).^(1 / γₑ) - (1. - γₑ) .* μ_K .+
                γₑ * (1. - γₑ) / 2 * σ^2 - σ_vₑ .* ((1 - γₑ) * σ)
            G_vₕ  = (ρₕ + νₕ) - ((1. .- η) .* q).^(1 / γₕ - 1.) ./ (vₕ.^(1 / γₕ)) - (1. - γₕ) .* μ_K .+
                γₕ * (1. - γₕ) / 2 * σ^2 - σ_vₕ .* ((1 - γₕ) * σ)
            err_vₑ = (μ_vₑ - G_vₑ) .* vₑ
            err_vₕ = (μ_vₕ - G_vₕ) .* vₕ

            return (G_vₑ, G_vₕ), vcat(err_vₑ, err_vₕ)
        end

        return f
    else
    end
end

function create_callback(m::BruSan, vₑ = nothing, vₕ = nothing)
    odecond = if m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
        function _ode_condition_brusan_log(q, η, integrator)
            cₑ_per_nₑ = m[:ρₑ] + m[:νₑ]
            cₕ_per_nₕ = m[:ρₕ] + m[:νₕ]
            φ_e = ((q * (m[:χ₂] * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1. - η)) + m[:χ₁]) - 1.) /
                   m[:χ₂] - m[:aₕ]) / (m[:aₑ] - m[:aₕ]) / η
            # φ_e = (q * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1 - η)) - m.aₕ) / (m.aₑ - m.aₕ) / η
            return φ_e * η - 1.
        end
    elseif m[:ψₑ].value == 1. / m[:γₑ].value && m[:ψₕ].value == 1. / m[:γₕ].value && !isnothing(vₑ) && !isnothing(vₕ)
        function _ode_condition_brusan_crra(q, η, integrator)
            cₑ_per_K = (η * q / vₑ(η))^(1 / m[:γₑ])
            cₕ_per_K = ((1 - η) * q / vₕ(η))^(1 / m[:γₕ])
            ι        = (m[:χ₁] * q - 1.) / m[:χ₂]
            φ_e_mul_η = ((cₑ_per_K + cₕ_per_K) + ι - m[:aₕ]) / (m[:aₑ] - m[:aₕ])

            return φ_e_mul_η - 1.
        end
    end

    ode_affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(odecond, ode_affect!)

    return cb
end


function _postprocessing_staticstep_brusan_nonunit_gammas!(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                                           derivs::OrderedDict{Symbol, Vector{S}},
                                                           endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple,
                                                           odesol::ODESolution, ode_f) where {S <: Real}
    # Set up
    @unpack ρₑ, ρₕ, γₑ, γₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ
    @unpack vₑ, vₕ, q = funcvar
    @unpack φ_e, φ_h, ςₑ, ςₕ, σ_q, μ_q, σ_η, μ_η, ι, σ_vₑ, σ_vₕ = endo
    @unpack ∂vₑ_∂η, ∂vₕ_∂η, ∂²vₑ_∂η², ∂²vₕ_∂η², ∂q_∂η, ∂²q_∂η² = derivs
    η = stategrid[:η]

    # Calculate capital allocations and price
    no1 = 1:findlast(η .<= odesol.t[end])
    is1 = (no1[end] + 1):length(η) # experts hold all capital
    q[no1] .= odesol(η[no1])
    ∂q_∂η[no1] .= map(j -> ode_f(q[j], θ, η[j]), no1)
    cₑ_per_nₑ = ρₑ + νₑ
    cₕ_per_nₕ = ρₕ + νₕ
    if cₑ_per_nₑ == cₕ_per_nₕ
        qₙ =  (χ₂ * aₑ + 1.) / (χ₂ * cₑ_per_nₑ + χ₁)
        q[is1]     .= qₙ
        ∂q_∂η[is1] .= 0.
    else
        q[is1] .= (χ₂ * aₑ + 1.) ./ (χ₂ .* (η[(i + 1):end] .* cₑ_per_nₑ +
                                            (1. .- η[(i + 1):end]) .* cₕ_per_nₕ) .+ χ₁)
        ∂q_∂η[is1] .= -(χ₂ * aₑ + 1) / (χ₂ * (η  * cₑ_per_nₑ + (1 - η) * cₕ_per_nₕ) + χ₁)^2 *
            (χ₂ * (cₑ_per_nₑ - cₕ_per_nₕ))
    end
    φ_e[no1] .= ((q[no1] .* (χ₂ .* (cₑ_per_nₑ .* η[no1] + cₕ_per_nₕ .* (1. .- η[no1])) .+ χ₁) .- 1.) ./ χ₂ .- aₕ) ./
        (aₑ - aₕ) ./ η[no1]
    φ_e[is1] .= 1. ./ η[is1]
    φ_h .= (1. .- φ_e .* η) ./ (1. .- η)

    # Calculate volatility and drift of η
    σ_vₑ_div_ssq = ∂vₑ_∂η ./ vₑ .* η .* (φ_e .- 1.) # σ_vₑ / (σ + σ_q)
    σ_vₕ_div_ssq = ∂vₕ_∂η ./ vₕ .* η .* (φ_e .- 1.)
    rp_div_ssq² = (γₑ .* φ_e - γₕ .* φ_h) +
        ((γₑ - 1.) .* σ_vₑ_div_ssq - (γₕ - 1.) .* σ_vₕ_div_ssq)  # (risk premium) / (σ + σ_q)²
    σ_q[no1] .= sqrt.(((aₑ - aₕ) ./ q[no1] ) ./ rp_div_ssq²[no1]) .- σ
    σ_q[is1] .= cₑ_per_nₑ == cₕ_per_nₕ ? 0. : ∂q_∂η[is1] .* η[is1] .* (φ_e[is1] .- 1.) .* σ ./
        (q[is1] .- ∂q_∂η[is1] .* η[is1] .* (φ_e[is1] .- 1.))
    σ_vₑ .= σ_vₑ_div_ssq .* (σ .+ σ_q)
    σ_vₕ .= σ_vₕ_div_ssq .* (σ .+ σ_q)
    σ_η  .= (φ_e .- 1.) .* (σ .+ σ_q)
    σ_η[1] = 0.
    σ_η[end] = 0.
    σ_vₑ[1] = 0.
    σ_vₕ[1] = 0.
    σ_vₑ[end] = 0.
    σ_vₕ[end] = 0.
    ςₑ   .= γₑ .* φ_e .* (σ .+ σ_q).^2 + (γₑ - 1.) .* σ_vₑ .* (σ .+ σ_q)
    ςₕ   .= γₕ .* φ_h .* (σ .+ σ_q).^2 + (γₕ - 1.) .* σ_vₕ .* (σ .+ σ_q)
    ι    .= (χ₁ .* q .- 1.) ./ χ₂
    μ_η  .= (aₑ .- ι) ./ q .- cₑ_per_nₑ .- τ .+
        (φ_e .- 1.) .* ((γₑ .* φ_e .- 1.) .* (σ .+ σ_q).^2 + (γₑ - 1.) .* σ_vₑ .* (σ .+ σ_q))
end

function _postprocessing_staticstep_brusan_crra!(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                                 derivs::OrderedDict{Symbol, Vector{S}},
                                                 endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple,
                                                 odesol::ODESolution, ode_f) where {S <: Real}
    # Set up
    @unpack ρₑ, ρₕ, γₑ, γₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = θ
    @unpack vₑ, vₕ, q = funcvar
    @unpack φ_e, φ_h, ςₑ, ςₕ, σ_q, μ_q, σ_η, μ_η, ι, σ_vₑ, σ_vₕ = endo
    @unpack ∂vₑ_∂η, ∂vₕ_∂η, ∂²vₑ_∂η², ∂²vₕ_∂η², ∂q_∂η, ∂²q_∂η² = derivs
    η = stategrid[:η]

    # Calculate capital allocations and price
    no1 = 1:findlast(η .<= odesol.t[end])
    is1 = (no1[end] + 1):length(η) # experts hold all capital
    q[no1] .= odesol(η[no1])
    ∂q_∂η[no1] .= map(j -> ode_f(q[j], θ, η[j]), no1)
    for i in is1
        q[i] = nlsolve(q -> (η[i] * q / vₑ[i]).^(1 / γₑ) + ((1. - η[i]) * q / vₕ[i]).^(1 / γₕ) +
                       (χ₁ * q .- 1.) / χ₂ .- aₑ, [q[i - 1]]).zero[1]
    end
    cₑ_per_K  = (η .* q ./ vₑ).^(1 / γₑ)
    cₕ_per_K  = ((1. .- η) .* q ./ vₕ).^(1 / γₕ)
    ι        .= (χ₁ * q .- 1.) / χ₂
    φ_e      .= ((cₑ_per_K + cₕ_per_K) + ι .- aₕ) ./ (aₑ - aₕ) ./ η
    φ_h      .= (1. - φ_e .* η) / (1. .- η)

    # Calculate volatility and drift of η
    σ_vₑ_div_ssq = ∂vₑ_∂η ./ vₑ .* η .* (φ_e .- 1.) # σ_vₑ / (σ + σ_q)
    σ_vₕ_div_ssq = ∂vₕ_∂η ./ vₕ .* η .* (φ_e .- 1.)
    rp_div_ssq²  = σ_vₕ_div_ssq - σ_vₑ_div_ssq + (φ_e .- 1.) ./ (1. .- η)
    σ_q[no1]    .= sqrt.(((aₑ - aₕ) ./ q[no1] ) ./ rp_div_ssq²[no1]) .- σ
    σ_q[is1]    .= ∂q_∂η[is1] .* η[is1] .* (φ_e[is1] .- 1.) .* σ ./
        (q[is1] .- ∂q_∂η[is1] .* η[is1] .* (φ_e[is1] .- 1.))
    σ_vₑ        .= σ_vₑ_div_ssq .* (σ .+ σ_q)
    σ_vₕ        .= σ_vₕ_div_ssq .* (σ .+ σ_q)
    σ_η         .= (φ_e .- 1.) .* (σ .+ σ_q)
    σ_η[1] = 0.
    σ_η[end] = 0.
    σ_vₑ[1] = 0.
    σ_vₕ[1] = 0.
    σ_vₑ[end] = 0.
    σ_vₕ[end] = 0.
    ςₑ  .= -σ_vₑ + σ_η + σ_q .+ γₑ * σ
    ςₕ  .= -σ_vₕ - η .* σ_η ./ (1. .- η) + σ_q .+ γₕ * σ
    μ_η .= (aₑ .- ι) ./ q .- cₑ_per_K ./ (η .* q) .- τ .+ σ_η .* (ςₑ - (σ .+ σ_q))
end
