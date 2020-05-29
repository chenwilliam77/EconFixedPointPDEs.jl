"""
```
initialize(m::Li2020)
```
sets up all initial conditions for solving Li2020, such as the grid and boundary conditions.
"""
function intiailize(m::Li2020)

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
    p₀ = find_zero(p -> get_setting(m, :Φ)(p) + m[:ρ] * p - m[:AL], 1.) # bisection search for p(0) and p(1)
    p₁ = find_zero(p -> get_setting(m, :Φ)(p) + m[:ρ] * p - m[:AH], 1.)
    set_boundary_conditions!(m, :p, [p₀.zero, p₁.zero])

    return stategrid, endo
end
