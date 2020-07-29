using Pkg
Pkg.activate(joinpath(dirname(@__FILE__), "../../../"))
using Test, OrdinaryDiffEq, HDF5, ModelConstructors, Plots, NLsolve, Statistics, BenchmarkTools
include(joinpath(dirname(@__FILE__), "../../../src/includeall.jl"))
include("fixedpoint_timestep.jl")

start_time = time_ns()

default = false
use_nlsolve = true # slower than default by .3 seconds, plus 2 extra steps
method = :newton
my_upwind = true
uniform = false
m_anderson = 3
bc = [(0., 1.)] # Vector{Tuple{Float64, Float64}}(undef, 0)

if default
    use_nlsolve = false
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
Δ  = .6
θ = (ρₑ = ρₑ, ρₕ = ρₕ, τ = τ, aₑ = aₑ, aₕ = aₕ, σ = σ, γ = γ, χ₁ = χ₁, χ₂ = χ₂, δ = δ, Δ = Δ)
pnorm = Inf
T = 100
tol = 1e-3
update_hyp = .8 # 1 is all weight on the ratio of a rolling window of error averages, 0 is none
reduce_hyp = .95 # proportion by which to reduce Δ if the error grows in size
discounting = 5

## Grid
# uneven grid on [0, 1] with 1000 points, with more points near 0 and 1

N   = 100 # 1000
zz  = collect(range(0.01, stop = 0.99, length = N))
# η   = zz.^1.1
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
# vₑ = 100 * η .^ (1. - γ)
# vₕ = 100 * (1. .- η) .^ (1. - γ)
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

#=println("Time Anderson acceleration")
@btime begin
out = nlsolve((F, x) -> fixedpoint_timestep(F, x, η, dη, Q, Qp, Psi, SSQ, MU, S, Svₑ, Svₕ, Mvₑ, Mvₕ, vₑ₁, vₕ₁, θ;
                                            use_nlsolve = use_nlsolve, method = method, my_upwind = my_upwind),
              vcat(vₑ, vₕ), method = :anderson, m = m_anderson, ftol = tol)
end=#
out = nlsolve((F, x) -> fixedpoint_timestep(F, x, η, dη, Q, Qp, Psi, SSQ, MU, S, Svₑ, Svₕ, Mvₑ, Mvₕ, vₑ₁, vₕ₁, θ;
                                            use_nlsolve = use_nlsolve, method = method, my_upwind = my_upwind,
                                            uniform = false),
              vcat(vₑ, vₕ), method = :anderson, m = m_anderson, ftol = tol)

N = length(η)
vₑ = out.zero[1:N]
vₕ = out.zero[(N + 1):end]
for ind in 2:(N - 1)
    Mvₑfd[ind] = ((vₑ[ind + 1] - vₑ[ind]) / dη[ind] * max(0, MU[ind]) +
                  (vₑ[ind] - vₑ[ind - 1]) / dη[ind - 1] * min(0, MU[ind])) / vₑ[ind] +
                  (vₑ[ind + 1] - 2 * vₑ[ind] + vₑ[ind - 1]) / (dη[ind] * dη[ind - 1]) / 2 / vₑ[ind] * S[ind]^2;
    Mvₕfd[ind] = ((vₕ[ind + 1] - vₕ[ind]) / dη[ind] * max(0, MU[ind]) +
                  (vₕ[ind] - vₕ[ind - 1]) / dη[ind - 1] * min(0, MU[ind])) / vₕ[ind] +
                  (vₕ[ind + 1] - 2 * vₕ[ind] + vₕ[ind - 1]) / (dη[ind] * dη[ind - 1]) / 2 / vₕ[ind] * S[ind]^2;
end
Mvₑfd[1] = ((vₑ[1 + 1] - vₑ[1]) / dη[1] * MU[1]) / vₑ[1]
Mvₕfd[1] = ((vₕ[1 + 1] - vₕ[1]) / dη[1] * MU[1]) / vₕ[1]
Mvₑfd[end] = (vₑ[end] - vₑ[end - 1]) / dη[end] * MU[end] / vₑ[end]
Mvₕfd[end] = (vₕ[end] - vₕ[end - 1]) / dη[end] * MU[end] / vₕ[end]


GS1 = (η        ./ vₑ ) .^ (1. / γ)
GS2 = ((1 .- η) ./ vₕ ) .^ (1. / γ)
CN  = GS1 .* Q .^ (1. / γ - 1.) ./ η # C/N*η
C_N = GS2 .* Q.^(1. / γ - 1.) ./ (1. .- η)
Mvₑ .= ρₑ .- CN  - (1 - γ) .* (Φ .- δ + Svₑ .* σ .- γ * σ^2 / 2.)
Mvₕ .= ρₕ .- C_N - (1 - γ) .* (Φ .- δ + Svₕ .* σ .- γ * σ^2 / 2.)
@show norm(vcat(Mvₑfd - Mvₑ, Mvₕfd - Mvₕ), pnorm)

end_time = time_ns()

elasped_time = (end_time - start_time) / 1e9


@show elasped_time


nothing
