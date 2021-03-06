"""
```
initialize!(m::Li2020)
```

sets up all initial conditions for solving Li2020, such as the grid and boundary conditions.
"""
function initialize!(m::Li2020)

    # Use no jump solution for p as initial guess
    stategrid, funcvar, derivs, endo = solve(m; nojump = true, nojump_method = :ode) # This calls initialize_nojump! already

    # Interpolate w/SLM, first over the "ODE" part, then interpolate once more
    p₀, p₁ = get_setting(m, :boundary_conditions)[:p]
    ψ_is_1 = findfirst(funcvar[:p] .>= p₁)
    p_SLM = SLM(stategrid[:w][1:ψ_is_1],  funcvar[:p][1:ψ_is_1];  concave_down = true, left_value = p₀,
                right_value = p₁, increasing = true, knots = floor(Int, ψ_is_1 / 4))
    funcvar[:p][1:ψ_is_1] .= eval(p_SLM, stategrid[:w][1:ψ_is_1])
    p_SLM = SLM(stategrid[:w], funcvar[:p], concave_down = true, increasing = true, left_value = p₀,
                right_value = p₁, knots = floor(Int, ψ_is_1 / 8))
    funcvar[:p] = eval(p_SLM, stategrid[:w])

    # Guess for xg and Q̂
    Q = get_setting(m, :gov_bond_gdp_level) * get_setting(m, :avg_gdp)
    funcvar[:xg]     .= Q ./ (2. .* stategrid[:w] .* funcvar[:p])
    funcvar[:xg][1]   = 2. * funcvar[:xg][2]
    funcvar[:xg][end] = m[:ϵ]
    funcvar[:Q̂]      .= fill(.05 * Q, length(stategrid))

    # Some settings for functional iteration
    m <= Setting(:κp_grid, vcat(0., exp.(range(log(1e-3), stop = log((p₁ - p₀) / p₁), length = 19))))

    return stategrid, funcvar, derivs, endo
end

"""
```
initialize_nojump!(m::Li2020)
```

initializes the no-jump equilibrium for the Li (2020) model.
"""
function initialize_nojump!(m::Li2020)
    model_type = eltype(m)
    N          = get_setting(m, :N)

    # Create StateGrid object
    stategrid_init = OrderedDict{Symbol, Vector{model_type}}()
    gen_grid(l, u, n) = exp.(range(log(l), stop = log(u), length = n))
    stategrid_init[:w] = vcat(0., gen_grid(get_setting(m, :stategrid_dimensions)[:w][1],
                                           get_setting(m, :stategrid_splice), Int(round(N / 2))),
                              gen_grid(get_setting(m, :stategrid_splice) + .01, get_setting(m, :stategrid_dimensions)[:w][2],
                              Int(N - round(N / 2) - 1)))
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

    # Establish boundary conditions
    p₀ = find_zero(p -> get_setting(m, :Φ)(p, m[:χ].value, m[:δ].value) + m[:ρ] * p - m[:AL], 1.) # bisection search for p(0) and p(1)
    p₁ = find_zero(p -> get_setting(m, :Φ)(p, m[:χ].value, m[:δ].value) + m[:ρ] * p - m[:AH], 1.)
    p₀ *= 1 + get_setting(m, :p₀_perturb)
    ∂p∂w0 = (m[:AH] - m[:AL]) / (get_setting(m, :∂Φ)(p₀, m[:χ].value) + m[:ρ]) *
                              ((m[:AH] - m[:AL]) / (p₀ * m[:σK])^2 * p₀ + 1)
    ∂p∂wN = 0.
    set_boundary_conditions!(m, :p, [p₀, p₁])
    set_boundary_conditions!(m, :∂p_∂w, [∂p∂w0, ∂p∂wN])

    return stategrid, funcvar, derivs, endo
end
