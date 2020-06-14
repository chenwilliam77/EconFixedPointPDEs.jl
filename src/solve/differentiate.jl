"""
```
```

The dict derivs map each variable to a number of derivative operators,
each of which should be applied to the diffvar dictionary.

I should also create another differentiate method that uses the same operators
for all the variables (so a higher level differentiate function).

You trigger this by adding a setting to your model object.

Use derivative order of the derivative operator to infer
the correct symbol name.
"""
function differentiate(stategrid::StateGrid, diffvar::Dict{Symbol, Vector{S}},
                       derivs::Dict{Symbol, Vector{AbstractDerivativeOperator}}) where {S <: Real}
    for (k, v) in derivs
        for state in keys(stategrid.x)
	        derivatives[Symbol(:∂, k :_∂, state)] = operators[:A] * v
        end
    end
end

"""
```
function differentiate(x::AbstractVector{S}, y::AbstractVector{S}) where {S <: Real}
```

first-order differentiates a variable using second-order central differences in the interior
and first-order forward/backward differences on the boundaries.

TO BE REPLACED BY DIFFEQOPERATOR LATER b/c this method is slow.
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
