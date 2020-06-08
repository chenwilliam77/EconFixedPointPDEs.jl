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
bin_sort(x::AbstractVector{T}, bins::AbstractVector{T};
    issorted::Bool = true) where {T <: Real}
```

returns a vector of Ints indicating in which bin each value of x belongs. The first function
assumes that `bins` defines the left edge of the bin and is sorted. If `x` is already sorted,
then we can speed up the calculation by taking advantage of `x` being sorted. The keyword
`issorted = true` indicates that `x` is sorted.
"""
function bin_sort(x::AbstractVector{S}, bins::AbstractVector{S};
                  issorted::Bool = true) where {S <: Real}

    # Fit the histogram
    h = fit(Histogram, x, bins)

    # Create index of bins
    if issorted
        nbins = length(bins)
        x_bins = zeros(S, nbins)
        ct = 1
        for i in 1:length(x)
            # Find index of the last point in each bin
            if x[i] >= bins[ct]
                x_bins[ct] = i - 1
                ct += 1

                if ct == nbins
                    # Then remaining points belong in the last bin
                    x_bins[ct] = length(x)

                    break
                end
            end
        end

        if max(x) == bins[end]
            # In this case, we lump max(x) with the previous bin
            x_bins = x_bins[1:end - 1]
            x_bins[end] += 1
        end

        # Create vector of bin numbers
        out = Vector{Int}(undef, length(x))
        out[1:x_bins[1]] = 1
        for i in 2:length(x_bins)
            out[x_bins[i - 1] + 1:x_bins[i]] = i
        end

        return out
    else
        out = map(y -> StatsBase.binindex(h, y), x)

        # Histogram uses [a, b) bins, but the last bin has just 1 point,
        # then we just lump it with the previous bin
        if max(x) == bins[end]
            max_vals = x .== max(x)
            out[max_vals] .= length(bins) - 1
        end

        return out
    end
end
