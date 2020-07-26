"""
```
```

The dictionary `diffop` map each variable to a number of derivative operators,
each of which should be applied to the `diffvar` dictionary.

I should also create another differentiate method that uses the same operators
for all the variables (so a higher level differentiate function).

You trigger this by adding a setting to your model object.

Use derivative order of the derivative operator to infer
the correct symbol name.
"""
function differentiate!(stategrid::StateGrid, diffvars::AbstractDict{Symbol, Vector{S}}, derivs::AbstractDict{Symbol, Vector{S}},
                        drift::Vector{S} = Vector{S}(undef, 0);
                        L₁ = nothing, L₂ = nothing, populate::Bool = false, uniform::Bool = false,
                        skipvar::Vector{Symbol} = Vector{Symbol}(undef, 0)) where {S <: Real}

    # Initialize finite difference operators with homogeneous Neumann boundary conditions
    if isnothing(L₁)
        drift = isempty(drift) ? vcat(ones(length(stategrid) - 1), -1.) : sign.(drift)

        # Default to a first-order forward finite difference w/backward finite difference at the end
        dx = uniform ? diff(values(stategrid.x)[1][1:2]) :
            vcat(values(stategrid.x[1])[1], diff(values(stategrid.x)[1]), diff(values(stategrid.x)[1][end - 1:end]))
        L₁ = if uniform
            UpwindDifference(1, 1, dx, length(stategrid), drift) *
                RobinBC((0., 1., 0.), (0., 1., 0.), dx)
        else
            UpwindDifference(1, 1, dx, length(stategrid), drift) *
                RobinBC((0., 1., 0.), (0., 1., 0.), dx)
        end
    else
        drift = isempty(drift) ? sign.(L₁.L.coefficients) : sign.(drift)
    end
    if isnothing(L₂)
        # Default to a second-order centered finite difference
        dx = uniform ? diff(values(stategrid.x)[1][1:2]) :
            vcat(stategrid.x[1], diff(values(stategrid.x)[1]), diff(stategrid.x[end - 1:end]))
        L₂ = CenteredDifference(2, 2, dx, length(stategrid)) * RobinBC((0., 1., 0.), (0., 1., 0.), dx)
    end

    # Differentiate all requested derivatives
    state_name = keys(stategrid.x)[1] # derivs maps to a vector ⇒ one-dimensional model
    for (k, v) in diffvars
        if k in skipvar
            continue
        end
        deriv1 = Symbol(:∂, k, :_∂, state_name)
        deriv2 = Symbol(:∂², k, :_∂, state_name, Symbol("²"))

        # Check if first derivative is in derivs or requested by populate
        if haskey(derivs, deriv1)
            derivs[deriv1] .= (drift * L₁) * v
        elseif populate
            derivs[deriv1]  = (drift * L₁) * v
        end

        # Check if second derivative is in derivs or requested by populate
        if haskey(derivs, deriv2)
            derivs[deriv2] .= L₂ * v
        elseif populate
            derivs[deriv2]  = L₂ * v
        end
    end
end

#=# Add another version of the above function with a dict that which maps variables to drifts that informs the DiffEqOperator
function differentiate(stategrid::StateGrid, diffvar::AbstractDict{Symbol, Vector{S}},
                       diffop::AbstractDict{Symbol, Vector{DiffEqOperators.AbstractDerivativeOperator}}) where {S <: Real}
    for (k, v) in diffop
        for state in keys(stategrid.x)
	        diffvar[Symbol(:∂, k :_∂, state)] = diffop[:A] * v
        end
    end
end
=#

"""
```
function differentiate(x::AbstractVector{S}, y::AbstractVector{S}) where {S <: Real}
```

first-order differentiates a variable using second-order central differences in the interior
and first-order forward/backward differences on the boundaries.
"""
function differentiate(x::AbstractVector{S}, y::AbstractVector{S}) where {S <: Real}

    # Set up
    N        = length(x)
    y_left   = y[1:(N - 2)]
    y_right  = y[3:N]
    y_middle = y[2:(N - 1)]
    x_left   = x[1:(N - 2)]
    x_right  = x[3:N]
    x_middle = x[2:(N - 1)]
    yx       = Vector{S}(undef, N)

    # Differentiate
    yx[2:(N - 1)] = (x_right - x_middle) ./ (x_right-x_left) .* (y_middle - y_left) ./ (x_middle - x_left) +
        (x_middle - x_left) ./ (x_right-x_left) .* (y_right - y_middle) ./ (x_right - x_middle)
    yx[1]   = (y[2] - y[1]) / (x[2] - x[1] );
    yx[end] = (y[end] - y[end - 1] ) / ( x[end] - x[end - 1])

    return yx
end
