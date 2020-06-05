"""
```
choose_knots(k, x)
```

calculates k uniformly spaced knots given a vector of grid points x
"""
function choose_knots(k::Int, x::AbstractVector{S}) where {S <: Real}
    return collect(range(min(x), stop = max(x), length = k))
end

"""
```
bin_sort(x, bins)
```

returns a BitArray indicating in which bin each value of x belongs. This function
assumes that x is in an ascending order.
"""
function bin_sort(x::AbstractVector{S}, bins::AbstractVector{S}) where {S <: Real}

    # Fit the histogram
    h = fit(Histogram, x, knots)
    x_bins = cumsum(h.weights)
    if max(x) == bins[end]
        x_bins[end] += 1 # Histogram will not count max(x) as part of the last bin, but for our purposes, we will
    end

    # Create vector of bin numbers
    out = Vector{Int}(undef, length(x))
    out[1:x_bins[1]] = 1
    for i in 2:length(x_bins)
        out[x_bins[i - 1] + 1:x_bins[i]] = i
    end

    return out
end
