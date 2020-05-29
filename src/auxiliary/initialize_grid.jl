"""
```
initialize_stategrid(method::Symbol, dims::Vector{Int})
```
constructs a state grid (using the implementation in EconPDEs)
using various methods, such as Chebyshev points.
"""
function initialize_stategrid(method::Symbol, grid_info::OrderedDict{Symbol, Tuple{Float64, Float64, Int}};
                              get_stategrid::Bool = true) where {S <: Real}
    stategrid_init = OrderedDict{Symbol, Vector{S}}()
    if method == :uniform
        for (k, v) in grid_info
            stategrid_init[k] = range(v[1], stop = v[2], length = v[3])
        end
    elseif method == :chebyshev
        for (k, v) in grid_info
            stategrid_init[k] = v[1] .+ 1. / 2. * (v[2] - v[1]) .* (1. .- cos(pi * (0:(v[3] - 1))' / (v[3] - 1)))
        end
    elseif method == :exponential
        for (k, v) in grid_info
            stategrid_init[k] = exp.(range(log(v[1]), stop = log(v[2]), length = v[3]))
        end
    elseif method == :smolyak
        error("Construction of a Smolyak interpolation grid has not been implemented yet.")
        # This should make a call to SmolyakApprox
    else
        error("Grid construction method $method has not been implemented.")
    end

    if get_stategrid
        return StateGrid(stategrid_init)
    else
        return stategrid_init
    end
end
