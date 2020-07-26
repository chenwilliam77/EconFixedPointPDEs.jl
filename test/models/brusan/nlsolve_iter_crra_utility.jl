using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), "../../../"))
using Test, OrdinaryDiffEq, HDF5, ModelConstructors, Plots, NLsolve, Statistics
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))

start_time = time_ns()

default = false
nlsolve_trustreg = false # slower than default by .3 seconds, plus 2 extra steps
nlsolve_newton = true  # roughly save time as trustreg
forwarddiff = false # Plus .6 seconds for newton, .75 for trust region
my_upwind = true
banded = false
uniform = true
bc = [(0., 1.)] # Vector{Tuple{Float64, Float64}}(undef, 0)

if default
    nlsolve_trustreg = false
    nlsolve_newton = false
    forwarddiff = false
end

## Parameters
ρₑ = 0.03
ρₕ = 0.03
τ  = .005
aₑ = .15
aₕ = .7 * aₑ
σ  = .03
γ  = 2.
χ₁ = 1.
χ₂ = 5.
δ  = 0.03
Δ  = .1
pnorm = Inf
T = 100
tol = 1e-6
update_hyp = .8 # 1 is all weight on the ratio of a rolling window of error averages, 0 is none
reduce_hyp = .95 # proportion by which to reduce Δ if the error grows in size
discounting = 5

## Grid
# uneven grid on [0, 1] with 1000 points, with more points near 0 and 1

N   = 150 # 1000
zz  = collect(range(0.001, stop = 0.999, length = N))
# η   = 3*zz.^2  - 2*zz.^3
η   = zz
dη  = diff(η)
stategrid = StateGrid(OrderedDict(:η => η))

if uniform
    L₁ = UpwindDifference(1, 1, dη[1], N)
    L₂ = CenteredDifference(2, 2, dη[1], N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), dη[1])
else
    dx = nonuniform_grid_spacing(stategrid, bc)
    L₁ = UpwindDifference(1, 1, dx, N)
    L₂ = CenteredDifference(2, 2, dx, N)
    Qbc = RobinBC((0., 1., 0.), (0., 1., 0.), dx)
end

default_err = fill(NaN, N, 3)
nlsolve_newton_err = fill(NaN, N, 3)
nlsolve_trustreg_err = fill(NaN, N, 3)

## Terminal conditions for value functions V and Vₕ
vₑ = aₑ^(-γ) .* η .^ (1. - γ)
vₕ = aₑ^(-γ) .* (1. .- η) .^ (1. - γ)
vₑ₀ = similar(vₑ)
vₕ₀ = similar(vₕ)
vₑ₁ = similar(vₑ)
vₕ₁ = similar(vₕ)
Q   =  ones(N)
Qp  = zeros(N)
SSQ = zeros(N)
Psi = zeros(N)
Iota   = zeros(N)
Φ   = zeros(N)
A   = zeros(N)
S   = zeros(N)
MU  = zeros(N)
Mvₑ  = zeros(N)
Mvₕ = zeros(N)
Mvₑfd  = zeros(N)
Mvₕfd = zeros(N)
G   = zeros(N)
Svₑ  = zeros(N)
Svₕ = zeros(N)
CN  = zeros(N)
C_N = zeros(N)

F = zeros(3) # For static step
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

t_err_vec = fill(NaN, T)
Δ_vec = fill(NaN, T)
dobreak = Bool[false]
Δ_vec[1:2] .= Δ
for t in 1:T

    ## Initialization
    GS1 = (η        ./ vₑ  ) .^ (1. / γ)
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
        if nlsolve_newton || nlsolve_trustreg
            EN = if forwarddiff
                nlsolve((F, x) -> staticstep!(F, x, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1), xₙ₋₁,
                         method = nlsolve_newton ? :newton : :trust_region, autodiff = :forward).zero
            else
                nlsolve((F, x) -> staticstep!(F, x, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1), xₙ₋₁,
                         method = nlsolve_newton ? :newton : :trust_region).zero
            end

            staticstep!(F, EN, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1)
            if nlsolve_newton
                nlsolve_newton_err[n, :] .= F
            elseif nlsolve_trustreg
                nlsolve_trustreg_err[n, :] .= F
            end
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
            staticstep!(F, EN, η[n], dη[n - 1], q_old, GS[n], VVlp[n - 1], 1)
            default_err[n, :] .= F
        end


#=            use_nlsolve_newton = false
            use_nlsolve_trustreg = false
            use_nlsolve_newton_qonly = false
            use_nlsolve_trustreg_qonly = false=#

        # if the boundary of the crisis regime has been reached, we have
        # ψ = 1 from now on
        if EN[2] > 1.
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
        q = Q[n - 1]

        ER = log(q) / γ + log(GS[n]) - log(aₑ - (χ₁ * q - 1.) / χ₂)
        QN = 1. / (q * γ)  + 1. / (aₑ - (q - 1.) / χ₂) / χ₂
        EN = q - ER / QN

        Qp[n]  = (EN - q) / dη[n-1]
        Q[n]   =  EN
        Psi[n] = 1.
        SSQ[n] = 1. / (1. - (1. - η[n]) * Qp[n] / Q[n]) * σ
    end

    ## Computing the PDE
    # Volatility of η
    S     = (Psi - η) .* SSQ # This is σ_η * η
    S[1]  = 0.
    S[N]  = 0.
    Iota .= (χ₁ .* Q .- 1.) ./ χ₂
    Φ    .= log.(χ₁ .* Q) / χ₂
    A    .= aₑ .* Psi + aₕ .* (1. .- Psi) - Iota

    # for the law of motion of η, C/N = (η Q)^(1/γ - 1)/V^(1/γ)
    # recall that GS1 = (η./V).^(1/γ)
    CN  = GS1 .* Q .^ (1. / γ - 1.) ./ η # C/N*η
    C_N = GS2 .* Q.^(1. / γ - 1.) ./ (1. .- η)

    Svₑ[2:N]  = vₑpl .* S[2:N]
    Svₕ[2:N] = vₕpl .* S[2:N]

    VarSig  = -Svₑ + S ./ η        + SSQ .+ (γ - 1) * σ
    VarSig_ = -Svₕ + S ./ (1. .- η) + SSQ .+ (γ - 1) * σ

    ind = 2:(N - 1)
    MU[ind] .= ((aₑ .- Iota[ind]) ./ Q[ind] - CN[ind] .- τ) .* η[ind] + S[ind] .* (VarSig[ind] - SSQ[ind]) # This is μ_η * η

    Mvₑ = ρₑ .- CN  - (1 - γ) .* (Φ .- δ + Svₑ .* σ .- γ * σ^2 / 2.)
    Mvₕ = ρₕ .- C_N - (1 - γ) .* (Φ .- δ + Svₕ .* σ .- γ * σ^2 / 2.)

    for ind in 2:(N - 1)
        Mvₑfd[ind] = ((vₑ[ind + 1] - vₑ[ind]) / dη[ind] * max(0, MU[ind]) +
            (vₑ[ind] - vₑ[ind - 1]) / dη[ind - 1] * min(0, MU[ind])) / vₑ[ind] +
            (vₑ[ind + 1] - 2 * vₑ[ind] + vₑ[ind - 1]) / (dη[ind] * dη[ind - 1]) / 2 / vₑ[ind] * S[ind]^2;
        Mvₕfd[ind] = ((vₕ[ind + 1] - vₕ[ind]) / dη[ind] * max(0, MU[ind]) +
            (vₕ[ind] - vₕ[ind - 1]) / dη[ind - 1] * min(0, MU[ind])) / vₕ[ind] +
            (vₕ[ind + 1] - 2 * vₕ[ind] + vₕ[ind - 1]) / (dη[ind] * dη[ind - 1]) / 2 / vₕ[ind] * S[ind]^2;
    end

    t_err_vec[t] = norm(vcat(Mvₑfd[2:end - 1] - Mvₑ[2:end - 1], Mvₕfd[2:end - 1] - Mvₕ[2:end - 1]), pnorm)

    ## Updating vₑ and vₕ
    # last parameter is dt*ρₑ if dt is small, can be at most 1 (1 corresponds to policy iteration)
    # it is more agressive to set the last parameter closer to 1, but code may not converge

    if t_err_vec[t] < tol
        break
    end

    if my_upwind
        pseudo_transient_relaxation!(stategrid, (vₑ₁, vₕ₁), (vₑ, vₕ), (Mvₑ, Mvₕ),
                                     MU ./ η, (S ./ η).^2, Δ_vec[t]; uniform = uniform, banded = banded,
                                     L₁ = L₁, L₂ = L₂, Q = Qbc)
    else
        vₑ₁ .= upwind_parabolic_pde(η, Mvₑ, MU, S.^2, G, vₑ, Δ_vec[t])
        vₕ₁ .= upwind_parabolic_pde(η, Mvₕ, MU, S.^2, G, vₕ, Δ_vec[t])
    end

    if t > 1
        if t_err_vec[t] / t_err_vec[t - 1] < 1.
            Δ_weight = (1 - update_hyp) + update_hyp * mean(t_err_vec[max(2, t - discounting):t] ./ t_err_vec[max(1, t - discounting - 1):t - 1])
            if t < T
                Δ_vec[t + 1] = (1 - Δ_weight) + Δ_weight * Δ_vec[t]
            end

            vₑ₀ .= vₑ
            vₕ₀ .= vₕ
            vₑ .= vₑ₁
            vₕ .= vₕ₁
        elseif t_err_vec[t] / t_err_vec[t - 1] > 1
            if t < T
                Δ_vec[t + 1] = reduce_hyp * Δ_vec[t]
            end
            vₑ .= vₑ₀
            vₕ .= vₕ₀
        end
    else
        vₑ .= vₑ₁
        vₕ .= vₕ₁
    end
end

end_time = time_ns()

elasped_time = (end_time - start_time) / 1e9
@show elasped_time

breakT = findfirst(isnan.(t_err_vec))
if isnothing(breakT)
    breakT = T
else
    breakT -= 1
    t_err_vec = t_err_vec[1:breakT]
    Δ_vec = Δ_vec[1:breakT]
end
@show t_err_vec[end]
@show breakT


nothing
