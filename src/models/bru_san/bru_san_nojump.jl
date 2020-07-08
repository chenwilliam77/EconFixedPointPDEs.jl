using EconPDEs, Distributions, NLsolve, UnPack

# This script implements an extension of the model presented
# in the Handbook of Macro Vol. 2 chapter
# "Macro, Money, and Finance: A Continuous Time Approach"
# by Brunnermeier and Sannikov.

# The extension includes agent-specific risk aversion, EIS,
# discount rates, death rates, and consumption good productivity.
# The extension, however, does not have any equity sharing between
# experts and households.

mutable struct BrunnermeierSannikov2016Model
    # Utility Function
    γₑ::Float64
    γₕ::Float64
    ψₑ::Float64
    ψₕ::Float64
    ρₑ::Float64
    ρₕ::Float64
    νₑ::Float64
    νₕ::Float64

    # Technology
    aₑ::Float64
    aₕ::Float64
    σ::Float64
    δ::Float64
    χ₁::Float64
    χ₂::Float64

    # Transfers
    τ::Float64
end

function BrunnermeierSannikov2016Model(; γₑ = 2., γₕ = 2., ψₑ = 2.,
                                       ψₕ = 2., ρₑ = .02, ρₕ = .02,
                                       νₑ = 0.01, νₕ = 0.,n
                                       aₑ = .1, aₕ = .07, σ = 0.03,
                                       δ = 0.0, χ₁ = 1., χ₂ = 1., τ = 0.)
    BrunnermeierSannikov2016Model(γₑ, γₕ, ψₑ, ψₕ,
                                  ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ,
                                  σ, δ, χ₁, χ₂, τ)
end

function initialize_stategrid(m::BrunnermeierSannikov2016Model; η_n = 80)
  OrderedDict(:η => range(0.01, stop = 0.99, length = η_n))
end

function initialize_y(m::BrunnermeierSannikov2016Model, stategrid::OrderedDict)
    x = fill(1.0, length(stategrid[:η]))
    cₑ_per_nₑ = (m.ψₑ == 1.) ? m.ρₑ + m.νₑ : x.^(-1)
    cₕ_per_nₕ = (m.ψₕ == 1.) ? m.ρₕ + m.νₕ : x.^(-1)
    q = nlsolve(q -> (m.aₑ - (χ₁ * q - 1) / χ₂ ) .- q .* (cₑ_per_nₑ .* x + cₕ_per_nₕ .* (1 .- x)), x ./ m.ρₑ).zero
    OrderedDict(:vₑ => x,
                :vₕ => x,
                :q => q)
end

function (m::BrunnermeierSannikov2016Model)(state::NamedTuple, y::NamedTuple)
    @unpack γₑ, γₕ, ψₑ, ψₕ, ρₑ, ρₕ, νₑ, νₕ, aₑ, aₕ, σ, δ, χ₁, χ₂, τ = m

    η = state[:η]
    @unpack vₑ, vₑη, vₑηη, vₕ, vₕη, vₕηη, q, qη, qηη = y

    # Determine leverage from market-clearing for capital and consumption
    #=
       aₑ φ_e * η + aₕ * φ_h * (1. - η)
     = aₑ φ_e * η + aₕ * (1 - φ_e * η) / (1 - η) * (1. - η)
     = aₑ φ_e * η + aₕ * (1 - φ_e * η)
     = aₕ + (aₑ - aₕ) * φ_e * η
    =#
    cₑ_per_nₑ = (ψₑ == 1.) ? ρₑ + νₑ : 1. / vₑ
    cₕ_per_nₕ = (ψₕ == 1.) ? ρₕ + νₕ : 1. / vₕ
    φ_e = ((q * (χ₂ * (cₑ_per_nₑ * η + cₕ_per_nₕ * (1. - η)) + χ₁) - 1.) / χ₂ - aₕ) / (aₑ - aₕ) / η
    φ_h = (1. - φ_e * η) / (1. - η)

    # Calculate capital-related quantities
    ι = (χ₁ * q - 1.) / χ₂
    Φ = χ₁ / χ₂ * log(χ₁ * q)

    # Compute volatilities
    debt_share = η * (φ_e - 1.) # Share of aggregate wealth held in debt
    σ_q  = qη * debt_share * σ / (q - qη * debt_share)
    σ_η  = (φ_e - 1) * (σ + σ_q)
    ησ_η = η * σ_η
    σ_vₑ = vₑη / vₑ * ησ_η
    σ_vₕ = vₕη / vₕ * ησ_η

    # Compute risk premia
    ςₑ = if ψₑ == 1.
        γₑ * φ_e * (σ + σ_q)^2 - (γₑ - 1.) * σ_vₑ * (σ + σ_q)
    else
        γₑ * φ_e * (σ + σ_q)^2 - (γₑ - 1.) / (ψₑ - 1.) * σ_vₑ * (σ + σ_q)
    end
    ςₕ = if ψₑ == 1.
        γₕ * φ_h * (σ + σ_q)^2 - (γₕ - 1.) * σ_vₕ * (σ + σ_q)
    else
        γₕ * φ_h * (σ + σ_q)^2 - (γₕ - 1.) / (ψₕ - 1.) * σ_vₕ * (σ + σ_q)
    end

    # Compute drifts
    μ_η  = (aₑ - ι) / q - cₑ_per_nₑ - τ +
        (φ_e - 1.) * ((γₑ * φ_e - 1.) * (σ + σ_q)^2 + (γₑ - 1.) / (ψₑ - 1.) * σ_vₑ * (σ + σ_q))
    ημ_η = η * μ_η
    μ_q  = qη / q * ημ_η + .5 * qηη / q * (ησ_η)^2
    μ_vₑ = vₑη / vₑ * ημ_η + .5 * vₑηη / vₕ * (ησ_η)^2
    μ_vₕ = vₕη / vₕ * ημ_η + .5 * vₑηη / vₕ * (ησ_η)^2

    # Compute interest rate
    dr_f = (aₑ - ι) / q + μ_q + Φ - δ + σ * σ_q - ςₑ

    # Evolution of marginal value of agents' net worth
    cₑ_term = (ψₑ == 1.) ? (ρₑ + νₑ) * log(cₑ_per_nₑ / vₑ) :
        ψₑ * ((cₑ_per_nₑ / vₑ^(1. / (ψₑ - 1.)))^(1. - 1. / ψₑ) - (ρₑ + νₑ))
    cₕ_term = (ψₕ == 1.) ? (ρₕ + νₕ) * log(cₕ_per_nₕ / vₕ) :
        ψₕ * ((cₕ_per_nₕ / vₕ^(1. / (ψₕ - 1.)))^(1. - 1. / ψₕ) - (ρₕ + νₕ))
    μ_Nₑ    = dr_f - cₑ_per_nₑ + φ_e * ςₑ
    μ_Nₕ    = dr_f - cₕ_per_nₕ + φ_h * ςₕ

    ∂vₑ_∂t = if ψₑ == 1.
        cₑ_term + μ_Nₑ - γₑ / 2 * (σ_vₑ^2 + (φ_e * (σ + σ_q))^2) + (1 - γₑ) * σ_vₑ * φ_e * (σ + σ_q)
    else
        cₑ_term + (ψₑ - 1.) * μ_Nₑ + μ_vₑ + .5 * (ψₑ - γₑ) / (ψₑ - 1.) * σ_vₑ^2 +
            (1. - γₑ) * σ_vₑ * φ_e * (σ + σ_q) - γₑ / 2. * (ψₑ - 1.) * (φ_e * (σ + σ_q))^2
    end

    ∂vₕ_∂t = if ψₕ == 1.
        cₕ_term + μ_Nₕ - γₕ / 2 * (σ_vₕ^2 + (φ_h * (σ + σ_q))^2) + (1 - γₕ) * σ_vₕ * φ_h * (σ + σ_q)
    else
        cₕ_term + (ψₕ - 1.) * μ_Nₕ + μ_vₕ + .5 * (ψ\_h - γₕ) / (ψₕ - 1.) * σ_vₕ^2 +
            (1. - γₕ) * σ_vₕ * φ_h * (σ + σ_q) - γₕ / 2. * (ψₕ - 1.) * (φ_h * (σ + σ_q))^2
    end

    # Asset pricing condition
    # if φ_e * η > 1, then we want it to decrease, so ∂q_∂t needs to decrease. Thus,
    # ∂q_∂t = 1 - φ_e * η, since this is negative when φ_e (and q) is too big.
    # If this doesn't work, then we'll have to do a nlsolve step for φ_e using an externally defined function
    ∂q_∂t = if φₑ * η > 1 # then this should multiply to 1
        1. - φ_e * η
    else # then both agents hold positive quantities of capital
        (aₑ - aₕ) / q - (ςₑ - ςₕ) # if this is positive, then φ_e is too small, so increase φ_e and q
    end

    # Return the negatives of the time derivatives, plus
    return (-∂vₑ_∂t, -∂vₕ_∂t, -∂q_∂t), (μ_η, ), (μ_η = μ_η, q = q, vₑ = vₑ, vₕ = vₕ,
                                                 φ_e = φ_e, φ_h = φ_h, dr_f = dr_f,
                                                 Φ = Φ, ι = ι, σ_q = σ_q,
                                                 σ_η = σ_η, σ_vₑ = σ_vₑ, σ_vₕ = σ_vₕ,
                                                 ςₑ = ςₑ, ςₕ = ςₕ)
end

m = BrunnermeierSannikov2016Model(ϵ₁ = 1., ϵ₂ = 1., γ₁ = 1., γ₂ = 1.)
stategrid = initialize_stategrid(m)
y0 = initialize_y(m, stategrid)
y, result, distance = pdesolve(m, stategrid, y0)
# y, result, distance = pdesolve(m, stategrid, y0; is_algebraic = OrderedDict(:vₑ => false, :vₕ => false, :q => true))
