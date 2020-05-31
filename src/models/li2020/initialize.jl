"""
```
initialize!(m::Li2020)
```
sets up all initial conditions for solving Li2020, such as the grid and boundary conditions.
"""
function initialize!(m::Li2020)

    model_type = eltype(m)
    N          = get_setting(m, :N)

    # Create StateGrid object
    stategrid_init = OrderedDict{Symbol, Vector{model_type}}()
    gen_grid(l, u, n) = exp.(range(log(l), stop = log(u), length = n))
    stategrid_init[:w] = vcat(0., gen_grid(get_setting(m, :stategrid_dimensions)[:w][1],
                                           get_setting(m, :stategrid_splice), Int(round(N / 2))),
                              gen_grid(get_setting(m, :stategrid_splice) + .1, get_setting(m, :stategrid_dimensions)[:w][2],
                              Int(N - round(N / 2) - 1)))
    stategrid = StateGrid(stategrid_init)

    # Construct dictionary of differential variables
    diffvar = OrderedDict{Symbol, Vector{model_type}}()
    for k in keys(get_differential_variables(m))
        diffvar[k] = Vector{model_type}(undef, N)
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
    set_boundary_conditions!(m, :∂p∂w, [∂p∂w0, ∂p∂wN])

    # Settings for functional iteration
    m <= Setting(:κp_guess, vcat(0., exp(range(log(1e-3), stop = log((p₁ - p₀) / p₁), length = 19))))

    return stategrid, diffvar, endo
end
