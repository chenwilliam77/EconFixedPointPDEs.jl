"""
```
init_subspec!(m::BruSan)
```
initializes alternative subspecs of the BruSan model
"""
function init_subspec!(m::BruSan)
    if subspec(m) == "ss0"
        return
    elseif subspec(m) == "ss1"
        ss1!(m)
    elseif subspec(m) == "ss2"
        ss2!(m)
    elseif subspec(m) == "ss3"
        ss3!(m)
    else
        error("Subspec $(subspec(m)) has not been defined.")
    end
end

"""
```
ss1!(m::BruSan)
```

initializes non-unit risk aversion specification of `BruSan`, with γₑ = γₕ = 2, but the EIS remains one.
"""
function ss1!(m::BruSan)
    m <= parameter(:γₑ, 2.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert risk aversion coefficient")
    m <= parameter(:γₕ, 2.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household risk aversion coefficient")

    m <= Setting(:stategrid_dimensions, OrderedDict{Symbol, Tuple{Float64, Float64, Int}}(:η => (1e-3, 1. - 1e-3, get_setting(m, :N))),
                 "Information about the dimensions of the state space")
end

"""
```
ss2!(m::BruSan)
```

initializes non-unit risk aversion, non-unit EIS specification of `BruSan`, with γₑ = γₕ = ψₑ = ψₕ = 2.
"""
function ss2!(m::BruSan)
    ss1!(m)
    m <= parameter(:ψₑ, 2.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert EIS")
    m <= parameter(:ψₕ, 2.0, (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household EIS")
end


"""
```
ss3!(m::BruSan)
```

initializes CRRA utility with γₑ = γₕ = 2.
"""
function ss3!(m::BruSan)
    ss1!(m)
    m <= parameter(:ψₑ, 1 / m[:γₑ], (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Expert EIS")
    m <= parameter(:ψₕ, 1 / m[:γₕ], (0., Inf), (0., 1e3), ModelConstructors.Untransformed(), Uniform(0., 1e3),
                   fixed = false, description = "Household EIS")
    m[:aₑ] = .15
    m[:aₕ] = .7 * m[:aₑ]
end
