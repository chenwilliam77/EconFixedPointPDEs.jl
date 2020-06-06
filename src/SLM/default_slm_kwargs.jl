"""
```
default_slm_kwargs!
```

adds the default keyword arguments for an SLM object.
"""
function default_slm_kwargs!(kwargs::Dict)

    # scaling
    if !haskey(kwargs, :scaling)
        kwargs[:scaling] = true
    end

    # knots
    if !haskey(kwargs, :knots)
        kwargs[:knots] = 6
    end

    # C2 function
    if !haskey(kwargs, :C2)
        kwargs[:C2] = true
    end

    # Regularization parameter
    if !haskey(kwargs, :λ)
        kwargs[:λ] = 1e-4
    end

    # Monotonicity
    if !haskey(kwargs, :increasing)
        kwargs[:increasing] = false
    end

    if !haskey(kwargs, :decreasing)
        kwargs[:decreasing] = false
    end

    if !haskey(kwargs, :increasing_intervals)
        kwargs[:increasing_intervals] = Matrix{Float64}(undef, 0, 0)
    end

    if !haskey(kwargs, :decreasing_intervals)
        kwargs[:decreasing_intervals] = Matrix{Float64}(undef, 0, 0)
    end

    # Curvature
    if !haskey(kwargs, :concave_up)
        kwargs[:concave_up] = false
    end

    if !haskey(kwargs, :concave_down)
        kwargs[:concave_down] = false
    end

    if !haskey(kwargs, :concave_up_intervals)
        kwargs[:concave_up_intervals] = Matrix{Float64}(undef, 0, 0)
    end

    if !haskey(kwargs, :concave_down_intervals)
        kwargs[:concave_down_intervals] = Matrix{Float64}(undef, 0, 0)
    end

    # End point values
    if !haskey(kwargs, :left_value)
        kwargs[:left_value] = NaN
    end

    if !haskey(kwargs, :right_value)
        kwargs[:right_value] = NaN
    end

    # Global min and max
    if !haskey(kwargs, :min_value)
        kwargs[:min_value] = NaN
    end

    if !haskey(kwargs, :max_value)
        kwargs[:max_value] = NaN
    end

    if !haskey(kwargs, :min_max_sample_points)
        kwargs[:min_max_sample_points] = [.017037, .066987, .1465, .25, .37059,
                                          .5, .62941, .75, .85355, .93301, .98296]
    end

    return kwargs
end
