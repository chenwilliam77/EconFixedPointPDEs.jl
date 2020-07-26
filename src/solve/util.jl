
"""
```
average_update(new::AbstractArray{S}, old::AbstractArray{S}, learning_rate::S) where {S <: Real}
```

updates `old` with `new` by taking a weighted average according to `learning_rate`.
"""
@inline function average_update(new::AbstractArray{S}, old::AbstractArray{S}, learning_rate::S) where {S <: Real}
    return learning_rate .* new + (1 - learning_rate) .* old
end

"""
```
function calculate_func_error(new_funcvar::OrderedDict, old_funcvar::OrderedDict, error_method::Symbol;
    vars::Vector{Symbol} = collect(keys(old_funcvar)))

function calculate_func_error(new_funcvar::AbstractArray{S}, old_funcvar::AbstractArray{S}, error_method::Symbol) where {S <: Real}
```

calculates the error between the proposal and the current guesses
for the functional variables. The keyword `vars` for the first method indicates which variables
we want to use to calculate the error (as sometimes, we want to ignore certain ones).
"""
@inline function calculate_func_error(new_funcvar::OrderedDict, old_funcvar::OrderedDict, error_method::Symbol;
                                      vars::Vector{Symbol} = collect(keys(old_funcvar)))

    if length(vars) == length(old_funcvar)
        new_vals = values(new_funcvar)
        old_vals = values(old_funcvar)
    else
        new_vals = map(x -> new_funcvar[x], vars)
        old_vals = map(x -> old_funcvar[x], vars)
    end

    func_error = if error_method == :total_error
        sum([sum(abs.(v1 - v2)) for (v1, v2)
             in zip(new_vals, old_vals)])
    elseif error_method in [:L∞, :Linf, :max_abs_error, :maximum_absolute_error]
        maximum([maximum(abs.(v1 - v2)) for (v1, v2)
                 in zip(new_vals, old_vals)])
    elseif error_method in [:L², :L2, :squared_error]
        sum([sum((v1 - v2) .^ 2) for (v1, v2)
             in zip(new_vals, old_vals)])
    end

    return func_error
end

@inline function calculate_func_error(new_funcvar::AbstractArray{S}, old_funcvar::AbstractArray{S}, error_method::Symbol) where {S <: Real}

    func_error = if error_method == :total_error
        sum(abs.(new_funcvar - old_funcvar))
    elseif error_method in [:L∞, :Linf, :max_abs_error, :maximum_absolute_error]
        maximum(abs.(new_funcvar - old_funcvar))
    elseif error_method in [:L², :L2, :squared_error]
        sum((new_funcvar - old_funcvar) .^ 2)
    end

    return func_error
end

"""
```
nojump_ode_init(m::AbstractNLCTFPModel)
```

sets up (by default) the initial condition for the no-jump solution via an ODE solver.
The inference may not always work.
"""
function nojump_ode_init(m::AbstractNLCTFPModel)
    @warn "Guessing the initial values for the no-jump ODE. It is recommended to overload the function `nojump_ode_init` " *
        "to make sure the correct initial values are used."
    bcs = get_setting(m, :boundary_conditions)
    out = map(x -> x[1], values(bcs))
    if length(out) == 1
        return out[1]
    else
        return out
    end
end

"""
```
function nonuniform_grid_spacing(stategrid::StateGrid, bc::AbstractVector{Tuple{S, S}} =
    Vector{Tuple{Float64, Float64}}(undef, 0)) where {S <: Real}
```

constructs the spacing vector for a non-uniform grid using a ghost-node approach.

If `bc` is empty, then we assume the spacing between the boundary and the boundary-adjacent grid points
are the same as the second most interior grid point. For a 1D grid `x`, this assumption means that
the spacing between the boundary and left/right endpoints are `diff(x)[1]` and `diff[x][end]`, respectively.

If the boundary values are known, then they can be passed to `bc` as a vector of the endpoints of
each dimension in a Cartesian product, e.g.
with a grid with boundaries `[a, b] × [c, d]`, the user would set `bc = [(a, b), (c, d)]`.
"""
function nonuniform_grid_spacing(stategrid::StateGrid, bc::AbstractVector{Tuple{S, S}} =
                                 Vector{Tuple{Float64, Float64}}(undef, 0)) where {S <: Real}
    @assert isempty(bc) || ndims(stategrid) == length(bc) "The dimensions of `stategrid` must match `bc`"
    if ndims(stategrid) == 1
        x = values(stategrid.x)[1]
        dx1 = isempty(bc) ? x[2] - x[1] : x[1] - bc[1][1]
        dxN = isempty(bc) ? x[end] - x[end - 1] : bc[1][2] - x[end]
        return vcat(dx1, diff(x), dxN)
    else
        error("Not yet implemented")
    end
end
