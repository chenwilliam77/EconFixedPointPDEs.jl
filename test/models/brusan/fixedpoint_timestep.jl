using UnPack
function staticstep!(F, x, η, dη, q_old, GS, VVlp, region)
    if region == 1
        q   = x[1]
        ψ   = x[2]
        ssq = x[3]

        F[1] = log(q) / γ + log(GS) - log(aₑ * ψ + aₕ * (1. - ψ) - (χ₁ * q - 1) / χ₂)
        F[2] = ssq * (q - (q - q_old) * (ψ - η) / dη) - σ * q
        F[3] = aₑ - aₕ - q * (ψ - η) * ssq^2 * VVlp
    elseif region == 2
        q    = x[1]
        F[1] = log(q) / γ + log(GS) - log(aₑ - (χ₁ * q - 1.) / χ₂)
    end
    nothing
end

function fixedpoint_timestep(F, vₑvₕ, η, dη, Q, Qp, Psi, SSQ, MU, S, Svₑ, Svₕ, Mvₑ, Mvₕ, vₑ₁, vₕ₁, θ;
                             use_nlsolve::Bool = false, method::Symbol = :newton, uniform::Bool = true,
                             my_upwind::Bool = false)
    @unpack ρₑ, ρₕ, τ, aₑ, aₕ, σ, γ, χ₁, χ₂, δ, Δ = θ
    N = length(η)

    vₑ = vₑvₕ[1:N]
    vₕ = vₑvₕ[(N + 1):end]
#= # For diagnosing problems. Typically boundary values becoming negative are the problem
    if any(vₑ .< 0.) || any(vₕ .< 0.)
        @show vₑ[1:5]
        @show vₑ[end - 4:end]
        @show vₕ[1:5]
        @show vₕ[end - 4:end]
        @show η[vₑ .< 0.]
        @show η[vₕ .< 0.]
    end
=#

    ## Initialization
    GS1 = (η        ./ vₑ ) .^ (1. / γ)
    GS2 = ((1 .- η) ./ vₕ ) .^ (1. / γ)
    GS  =  GS1 + GS2

    vₑp  = diff(vₑ)  ./ dη
    vₕp = diff(vₕ) ./ dη

    vₑpl  = vₑp  ./ vₑ[2:N]
    vₕpl = vₕp ./ vₕ[2:N]
    VVlp = vₕpl - vₑpl + 1. ./ (η[2:N] .* (1. .- η[2:N]))

    ## Initial condition at η[1]
    Psi[1] = 0.
    SSQ[1] = σ
    qL     = [0.]
    qR     = [aₑ * χ₂ + 1.]
    q₀     = [0.]
    for k = 1:30,
        q₀[1] = (qL[1] + qR[1]) / 2.
        ι     = (χ₁ * q₀[1] - 1.) / χ₂
        A₀   = aₕ - ι

        if log(q₀[1]) / γ + log(GS[1]) > log(A₀)
            qR[1] = q₀[1]
        else
            qL[1] = q₀[1]
        end
    end
    Q[1]  = q₀[1]

    ## Newton method
    #  find q, psi and ssq = σ + σ^q from value functions
    break_pt = Int[0]
    for n = 2:N
        q     = Q[n - 1]
        q_old = Q[n - 1]
        ψ     = Psi[n - 1]
        ssq   = SSQ[n - 1]

        xₙ₋₁  = [q, ψ, ssq]

        # errors given guess
        if use_nlsolve
            EN = nlsolve((F, x) -> staticstep!(F, x, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1), xₙ₋₁,
                         method = method).zero
        elseif default
            ER = [log(q) / γ + log(GS[n]) - log(aₑ * ψ + aₕ * (1. - ψ) - (χ₁ * q - 1) / χ₂),
                  ssq * (q - (q - q_old) * (ψ - η[n]) / dη[n - 1]) - σ * q,
                  aₑ - aₕ - q * (ψ - η[n]) * ssq^2 * VVlp[n - 1]]

            # matrix of derivatives of errors
            # could shorten it since q_old = q

            QN = zeros(3,3)

            QN[1, :] = [1 / (q * γ) + 1/((aₑ - aₕ) * ψ + aₕ - (q - 1.) / χ₂) / χ₂,
                        -(aₑ - aₕ) / ((aₑ - aₕ) * ψ + aₕ - (q - 1.) / χ₂), 0.]

            QN[2, :] = [ssq * (1. - (ψ - η[n]) / dη[n - 1]) - σ,
                        -ssq * (q - q_old) / dη[n-1],  q - (q - q_old) * (ψ - η[n]) / dη[n-1]]

            QN[3, :] = [- (ψ - η[n]) * ssq^2 * VVlp[n-1],
                        -q * ssq^2 * VVlp[n - 1], -2 * q * (ψ - η[n]) * ssq * VVlp[n - 1]]

            EN = [q, ψ, ssq] - QN \ ER
        end

        # if the boundary of the crisis regime has been reached, we have
        # ψ = 1 from now on
        if EN[2] > 1.
            if EN[1] < 0
                out = nlsolve((F, x) -> staticstep!(F, x, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1), xₙ₋₁,
                         method = method)
                @show out
            end
            break_pt[1] = n
            break
        end

        # save results
        Q[n]   = EN[1]
        Psi[n] = EN[2]
        SSQ[n] = EN[3]
        Qp[n]  = (Q[n] - q_old) / dη[n-1]
    end

    ## Newton method for ψ = 1, for remaining eta
    for n in break_pt[1]:N
        q = max(Q[n - 1], 1e-6)

        #=ER = log(q) / γ + log(GS[n]) - log(aₑ - (χ₁ * q - 1.) / χ₂)
        QN = 1. / (q * γ)  + 1. / (aₑ - (q - 1.) / χ₂) / χ₂
        EN = q - ER / QN
        EN = find_zero(x -> x^(1 / γ) * GS[n] - (aₑ - (χ₁ * x - 1.) / χ₂), (0., q * 2))=#
        EN = nlsolve(x -> x.^(1 / γ) .* GS[n] - (aₑ .- (χ₁ .* x .- 1.) ./ χ₂), [q], method = :newton).zero[1]

        Qp[n]  = (EN - q) / dη[n-1]
        Q[n]   =  EN
        Psi[n] = 1.
        SSQ[n] = 1. / (1. - (1. - η[n]) * Qp[n] / Q[n]) * σ
    end

    ## Computing the PDE
    # Volatility of η
    S    .= (Psi - η) .* SSQ # This is σ_η * η
    S[1]  = 0.
    S[N]  = 0.
    Iota .= (χ₁ .* Q .- 1.) ./ χ₂
    Φ    .= log.(χ₁ .* Q) / χ₂
    A    .= aₑ .* Psi + aₕ .* (1. .- Psi) - Iota

    # for the law of motion of η, C/N = (η Q)^(1/γ - 1)/V^(1/γ)
    # recall that GS1 = (η./V).^(1/γ)
    CN  = GS1 .* Q .^ (1. / γ - 1.) ./ η # C/N*η
    C_N = GS2 .* Q.^(1. / γ - 1.) ./ (1. .- η)

    Svₑ[2:N] .= vₑpl .* S[2:N]
    Svₕ[2:N] .= vₕpl .* S[2:N]

    VarSig  = -Svₑ + S ./ η        + SSQ .+ (γ - 1) * σ
    VarSig_ = -Svₕ + S ./ (1. .- η) + SSQ .+ (γ - 1) * σ

    ind = 2:(N - 1)
    MU[ind] .= ((aₑ .- Iota[ind]) ./ Q[ind] - CN[ind] .- τ) .* η[ind] + S[ind] .* (VarSig[ind] - SSQ[ind]) # This is μ_η * η

    Mvₑ .= ρₑ .- CN  - (1 - γ) .* (Φ .- δ + Svₑ .* σ .- γ * σ^2 / 2.)
    Mvₕ .= ρₕ .- C_N - (1 - γ) .* (Φ .- δ + Svₕ .* σ .- γ * σ^2 / 2.)

    ## Updating vₑ and vₕ
    # last parameter is dt*ρₑ if dt is small, can be at most 1 (1 corresponds to policy iteration)
    # it is more agressive to set the last parameter closer to 1, but code may not converge

    if my_upwind
        pseudo_transient_relaxation!(stategrid, (vₑ₁, vₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ; uniform = uniform)
    else
        vₑ₁ .= upwind_parabolic_pde(η, Mvₑ, MU, S.^2, G, vₑ, Δ)
        vₕ₁ .= upwind_parabolic_pde(η, Mvₕ, MU, S.^2, G, vₕ, Δ)
    end

    dt              = Δ / (1 - Δ)
    F[1:N]         .= (vₑ₁ - vₑ) ./ dt
    F[(N + 1):end] .= (vₕ₁ - vₕ) ./ dt
end
