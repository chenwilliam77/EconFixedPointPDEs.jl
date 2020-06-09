"""
```
choose_knots(k, x)
```

calculates k uniformly spaced knots given a vector of grid points x
"""
function choose_knots(k::Int, x::AbstractVector{S}) where {S <: Real}
    return collect(range(minimum(x), stop = maximum(x), length = k))
end

"""
```
bin_sort(x::AbstractVector{T}, bins::AbstractVector{T}) where {T <: Real}
```

returns a vector of Ints indicating in which bin each value of x belongs. The first function
assumes that `bins` defines the left edge of the bin and is sorted.
"""
function bin_sort(x::AbstractVector{S}, bins::AbstractVector{S}) where {S <: Real}

    # Fit the histogram
    h = fit(Histogram, x, bins)

    # Create index of bins
    out = map(y -> StatsBase.binindex(h, y), x)

    # Histogram uses [a, b) bins, but the last bin has just 1 point,
    # then we just lump it with the previous bin
    if maximum(x) == bins[end]
        max_vals = x .== maximum(x)
        out[max_vals] .= length(bins) - 1
    end

    return out
end

# """
# ```
# function accumarray(subs, val, sz=(maximum(subs),))
# ```

# See VectorizedRoutines.jl
# """
# function accumarray(subs, val, sz=(maximum(subs),))
#     A = zeros(eltype(val), sz...)
#     for i = 1:length(val)
#         @inbounds A[subs[i]...] += val[i]
#     end
#     A
# end
