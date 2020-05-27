"""
```
initialize(m::Li2020)
```
sets up all initial conditions for solving Li2020, such as the grid and boundary conditions.
"""
function intiailize(m::Li2020)
    endo = OrderedDict{Symbol, Vector{get_type(m)}}()

    for k in keys(m.endogenous_variables
end
