"""
```
initialize!(m::BruSan)
```

sets up all initial conditions for solving BruSan, such as the grid and boundary conditions.
"""
function initialize!(m::BruSan)

    stategrid, funcvar, derivs, endo = initialize_nojump!(m)


    #= # Use no jump solution for q as initial guess
    stategrid, funcvar, derivs, endo = solve(m; nojump = true, nojump_method = :ode) # This calls initialize_nojump! already

    # Interpolate w/SLM, first over the "ODE" part, then interpolate once more
     q₀, q₁ = get_setting(m, :boundary_conditions)[:q]
    ψ_is_1 = findfirst(funcvar[:p] .>= q₁)
    p_SLM = SLM(stategrid[:w][1:ψ_is_1],  funcvar[:p][1:ψ_is_1];  concave_down = true, left_value = q₀,
                right_value = q₁, increasing = true, knots = floor(Int, ψ_is_1 / 4))
    funcvar[:p][1:ψ_is_1] .= eval(p_SLM, stategrid[:w][1:ψ_is_1])
    p_SLM = SLM(stategrid[:w], funcvar[:p], concave_down = true, increasing = true, left_value = q₀,
                right_value = q₁, knots = floor(Int, ψ_is_1 / 8))
    funcvar[:p] = eval(p_SLM, stategrid[:w])

    # Guess for xg and Q̂
    Q = get_setting(m, :gov_bond_gdp_level) * get_setting(m, :avg_gdp)
    funcvar[:xg]     .= Q ./ (2. .* stategrid[:w] .* funcvar[:p])
    funcvar[:xg][1]   = 2. * funcvar[:xg][2]
    funcvar[:xg][end] = m[:ϵ]
    funcvar[:Q̂]      .= fill(.05 * Q, length(stategrid))

    # Some settings for functional iteration
    m <= Setting(:κp_grid, vcat(0., exp.(range(log(1e-3), stop = log((q₁ - q₀) / q₁), length = 19))))=#

    return stategrid, funcvar, derivs, endo
end

"""
```
initialize_nojump!(m::BruSan)
```

initializes the no-jump equilibrium for the Li (2020) model.
"""
function initialize_nojump!(m::BruSan)
    model_type = eltype(m)
    N          = get_setting(m, :N)

    # Create StateGrid object
    stategrid_init = OrderedDict{Symbol, Vector{model_type}}()
    gen_grid(l, u, n) = exp.(range(log(l), stop = log(u), length = n))
    stategrid_init[:η] = vcat(gen_grid(get_setting(m, :stategrid_dimensions)[:η][1],
                                           get_setting(m, :stategrid_splice), Int(round(N / 2))),
                              gen_grid(get_setting(m, :stategrid_splice) + .01, get_setting(m, :stategrid_dimensions)[:η][2],
                                       Int(N - round(N / 2))))
    zz  = collect(range(0.001, stop = 0.999, length = get_setting(m, :N)))
    stategrid_init[:η] = 3. .* zz .^ 2  - 2. .* zz .^ 3;

    stategrid = StateGrid(stategrid_init)

    # Construct dictionary of functional variables
    funcvar = OrderedDict{Symbol, Vector{model_type}}()
    for k in keys(get_functional_variables(m))
        funcvar[k] = Vector{model_type}(undef, N)
    end

    # Construct dictionary of derivatives
    derivs = OrderedDict{Symbol, Vector{model_type}}()
    for k in keys(get_derivatives(m))
        derivs[k] = Vector{model_type}(undef, N)
    end

    # Construct dictionary of endogenous variables
    endo = OrderedDict{Symbol, Vector{model_type}}()
    for k in keys(get_endogenous_variables(m))
        endo[k] = Vector{model_type}(undef, N)
    end

    if m[:γₑ].value != 1. && m[:γₕ].value != 1.
        qmin = (m[:χ₂] * m[:aₕ] + 1.) / (m[:χ₂] * (m[:ρₕ] + m[:νₕ]) + m[:χ₁])
        qmax = (m[:χ₂] * m[:aₑ] + 1.) / (m[:χ₂] * (m[:ρₑ] + m[:νₑ]) + m[:χ₁])
        qguess = (qmin + qmax) / 2.
        #=if m[:ψₑ].value == 1. && m[:ψₕ].value == 1.
            funcvar[:vₑ] .= (m[:aₑ] * qguess * (m[:γₑ] / (m[:γₑ] + m[:γₕ]))) .* stategrid[:η]
            funcvar[:vₕ] .= (m[:aₕ] * qguess * 2 * (m[:γₕ] / (m[:γₑ] + m[:γₕ]))) .* (1. .- stategrid[:η])
        else=#
            funcvar[:vₑ] .= ones(length(stategrid)) # (m[:aₑ] .* stategrid[:η]).^(m[:ψₑ] - 1.)
            funcvar[:vₕ] .= ones(length(stategrid)) # m[:aₕ] .* (1. .- stategrid[:η]).^(m[:ψₕ] - 1.)
        #end
    end

    # Establish boundary conditions
    if m[:ψₑ].value == 1.
        cₑ_per_nₑ = m[:ρₑ] + m[:νₑ]
    else
        cₑ_per_nₑ = 1. / funcvar[:vₑ][end]
    end
    if m[:ψₕ].value == 1.
        cₕ_per_nₕ = m[:ρₕ] + m[:νₕ]
    else
        cₕ_per_nₕ = 1. / funcvar[:vₕ][1]
    end

    q₀ = (m[:χ₂] * m[:aₕ] + 1) / (m[:χ₂] * cₕ_per_nₕ + m[:χ₁])
    q₁ = (m[:χ₂] * m[:aₑ] + 1) / (m[:χ₂] * cₑ_per_nₑ + m[:χ₁])
    q₀ *= 1 + get_setting(m, :q₀_perturb)
    set_boundary_conditions!(m, :q, [q₀, q₁])

    return stategrid, funcvar, derivs, endo
end
