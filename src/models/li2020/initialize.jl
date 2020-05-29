"""
```
initialize!(m::Li2020)
```
sets up all initial conditions for solving Li2020, such as the grid and boundary conditions.
"""
function initialize!(m::Li2020)

    # Create StateGrid object
    stategrid_init = initialize_stategrid(get_setting(m, :stategrid_method), get_setting(m, :stategrid_dimensions),
                                     get_stategrid = false)
    stategrid_init[:w] = vcat(0., stategrid_init[:w])
    stategrid = StateGrid(stategrid_init)

    # Construct dictionary of endogenous variables
    endo = OrderedDict{Symbol, Vector{get_type(m)}}()
    model_type = get_type(m)
    N          = get_setting(m, :N)
    for k in keys(m.endogenous_variables)
        endo[k] = Vector{model_type}(undef, N)
    end

    # Establish boundary conditions
    p₀ = find_zero(p -> get_setting(m, :Φ)(p, m[:χ].value, m[:δ].value) + m[:ρ] * p - m[:AL], 1.) # bisection search for p(0) and p(1)
    p₁ = find_zero(p -> get_setting(m, :Φ)(p, m[:χ].value, m[:δ].value) + m[:ρ] * p - m[:AH], 1.)
    ∂p∂w0 = (m[:AH] - m[:AL]) / (get_setting(m, :∂Φ)(p₀, m[:χ].value) + m[:ρ]) *
                              ((m[:AH] - m[:AL]) / (p₀ * m[:σK])^2 * p₀ + 1)
    ∂p∂wN = 0.
    set_boundary_conditions!(m, :p, [p₀, p₁])
    set_boundary_conditions!(m, :∂p∂w, [∂p∂w0, ∂p∂wN])

    return stategrid, endo
end
