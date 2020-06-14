# This file holds different types of investment functions
@inline function quadratic_investment(p::S, χ::S, δ::S) where {S <: Real}
    return (p .- 1.) .^ 2 ./ (2. .* χ) + (p .- 1.) ./ χ .+ δ
end

@inline function growth_quadratic_investment(p::S, χ::S, δ::S) where {S <: Real}
    return (p - 1.) / χ  + δ
end

@inline function derivative_quadratic_investment(p::S, χ::S) where {S <: Real}
    return p ./ χ + 1. ./ χ
end

@inline function derivative_quadratic_investment_li2020(p::S, χ::S) where {S <: Real}
    return p ./ χ
end
