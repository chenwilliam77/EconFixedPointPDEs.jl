# This file contains functions that create properties for the desired least-square spline

"""
```
default_slm_kwargs!(kwargs)
```

populates the kwargs dictionary with default values.
"""
function default_slm_kwargs!(kwargs::Dict{Symbol, Any})

    if !haskey(kwargs, :degree)
        kwargs[:degree] = 3
    end

    if !haskey(kwargs, :knots)
        kwargs[:knots] = 6
    end

    if !haskey(kwargs, :scaling)
        kwargs[:scaling] = on
    end

    if !haskey(kwargs, :C2)
        kwargs[:C2] = true
    end

    if !haskey(kwargs, :increasing)
        kwargs[:increasing] = false
    end

    if !haskey(kwargs, :decreasing)
        kwargs[:decreasing] = false
    end

    if !haskey(kwargs, :left_value)
        kwargs[:left_value] = NaN
    end

    if !haskey(kwargs, :right_value)
        kwargs[:right_value] = NaN
    end

    if !haskey(kwargs, :concave_up)
        kwargs[:concave_up] = false
    end

    if !haskey(kwargs, :concave_down)
        kwargs[:concave_down] = false
    end
end

"""
```
C2_matrix(nk::Int, nc::Int, dknots::AbstractVector{T}) where {T <: Real}
```

constructs the matrix of equations that enforce twice-continuous differentiability.
"""
function C2_matrix(nk::Int, nc::Int, dknots::AbstractVector{T}) where {T <: Real}
    MC2 = zeros(T, nk - 2, nc)
    for i in 1:(nk - 2)
        MC2[i, i] = 6. / dknots[i]^2
        MC2[i, i + 1] = -6. / dknots[i]^2 + 6. / dknots[i + 1]^2
        MC2[i, i + 2] = -6. / dknots[i + 1]^2

        MC2[i, nk + i] = 2. / dknots[i]
        MC2[i, nk + i + 1] = 4. / dknots[i] + 4. / dknots[i + 1]
        MC2[i, nk + i + 2] =  2. / dknots[i]
    end

    return MC2
end

"""
```
function monotone_increasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool, Vector{T}}}},
    nk::Int) where {T <: Real}
```

adds the properties of a monotone increasing spline.
"""
function monotone_increasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                              nk::Int) where {T <: Real}
    push!(monotone_settings, (knotlist = (1, nk), increasing = true, range = Vector{T}(undef, 0)))
    return nk - 1
end

"""
```
function increasing_intervals_info!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                   knots::AbstractVector{T}, increasing_intervals::AbstractMatrix{T}, nk::Int = length(knots)) where {T <: Real}
```

adds information about interval(s) over which the spline should be increasing
"""
function increasing_intervals_info!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                   knots::AbstractVector{T}, increasing_intervals::AbstractMatrix{T}, nk::Int = length(knots)) where {T <: Real}
    out = 0
    for i in 1:size(increasing_intervals, 1)
        knotlist = (max(1, findlast(knots .<= increasing_intervals[i, 1])),
                    min(nk, 1 + findlast(knots .< increasing_intervals[i, 2])))
        push!(monotone_settings, (knotlist = knotlist, increasing = true))
        out += diff(knotlist)
    end

    return out
end

"""
```
function monotone_decreasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
    nk::Int) where {T <: Real}
```

adds the properties of a monotone decreasing spline.
"""
function monotone_decreasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing, :range), Tuple{Tuple{Int, Int}, Bool}}},
                              nk::Int) where {T <: Real}
    push!(monotone_settings, (knotlist = (1, nk), increasing = false))
    return nk - 1
end

"""
```
function decreasing_intervals_info!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                   knots::AbstractVector{T}, decreasing_intervals::AbstractMatrix{T}, nk::Int = length(knots)) where {T <: Real}
```

adds information about interval(s) over which the spline should be decreasing
"""
function decreasing_intervals_info!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                   knots::AbstractVector{T}, decreasing_intervals::AbstractMatrix{T}, nk::Int = length(knots)) where {T <: Real}
    out = 0
    for i in 1:size(decreasing_intervals, 1)
        knotlist = (max(1, findlast(knots .<= decreasing_intervals[i, 1])),
                    min(nk, 1 + findlast(knots .< decreasing_intervals[i, 2])))
        push!(monotone_settings, (knotlist = knotlist, increasing = false))
        out += diff(knotlist)
    end

    return out
end

"""
```
function construct_monotoncitiy_matrix(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                      nc::Int, nk::Int, dknots::AbstractVector{T}, total_intervals::Int,
                                      Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T}) where {T <: Real}
```
The function must be monotone between
knots j and j + 1. The direction over
that interval is specified. The constraint
system used comes from Fritsch & Carlson, see here:

http://en.wikipedia.org/wiki/Monotone_cubic_interpolation

Define delta = (y(i+1) - y(i))/(x(i+1) - x(i))
Thus delta is the secant slope of the curve across
a knot interval. Further, define alpha and beta as
the ratio of the derivative at each end of an
interval to the secant slope.

 alpha = d(i)/delta
 beta = d(i+1)/delta

Then we have an elliptically bounded region in the
first quadrant that defines the set of monotone cubic
segments. We cannot define that elliptical region
using a set of linear constraints. However, by use
of a system of 7 linear constraints, we can form a
set of sufficient conditions such that the curve
will be monotone. There will be some few cubic
segments that are actually monotone, yet lie outside
of the linear system formed. This is acceptable,
as our linear approximation here is a sufficient
one for monotonicity, although not a necessary one.
It merely says that the spline may be slightly over
constrained, i.e., slightly less flexible than is
absolutely necessary. (So?)

The 7 constraints applied for an increasing function
are (in a form that lsqlin will like):

   -delta          <= 0
   -alpha          <= 0
   -beta           <= 0
   -alpha + beta   <= 3
    alpha - beta   <= 3
    alpha + 2*beta <= 9
    2*alpha + beta   <= 9

Multiply these inequalities by (y(i+1) - y(i)) to
put them into a linear form.
"""
function construct_monotoncitiy_matrix(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                                      nc::Int, nk::Int, dknots::AbstractVector{T}, total_intervals::Int,
                                      Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T}) where {T <: Real}
    L = length(monotone_settings)
    M = zeros(T, 7 * total_intervals, nc)
    n = 0

    for i in 1:L
        for j in monotone_settings[i][:knotlist][1]:(monotone_settings[i][:knotlist][2] - 1)
            if monotone_settings[i][:increasing]
                M[n + 1, j]          = 1.
                M[n + 1, j + 1]      = -1
                M[n + 2, nk + j]     = -1.
                M[n + 3, nk + j + 1] = -1.

                M[n + 4, j]          = 3.
                M[n + 4, j + 1]      = -3.
                M[n + 4, nk + j]     = -1. / dknots[j]
                M[n + 4, nk + j + 1] = 1. / dknots[j]

                M[n + 5, j]          = 3.
                M[n + 5, j + 1]      = -3.
                M[n + 5, nk + j]     = 1. / dknots[j]
                M[n + 5, nk + j + 1] = -1. / dknots[j]

                M[n + 6, j]          = 9.
                M[n + 6, j + 1]      = -9.
                M[n + 6, nk + j]     = 1. / dknots[j]
                M[n + 6, nk + j + 1] = 2. / dknots[j]

                M[n + 7, j]          = 9.
                M[n + 7, j + 1]      = -9.
                M[n + 7, nk + j]     = 2. / dknots[j]
                M[n + 7, nk + j + 1] = 1. / dknots[j]
            else
                M[n + 1, j]          = -1.
                M[n + 1, j + 1]      = 1
                M[n + 2, nk + j]     = 1.
                M[n + 3, nk + j + 1] = 1.

                M[n + 4, j]          = -3.
                M[n + 4, j + 1]      = 3.
                M[n + 4, nk + j]     = 1. / dknots[j]
                M[n + 4, nk + j + 1] = -1. / dknots[j]

                M[n + 5, j]          = -3.
                M[n + 5, j + 1]      = 3.
                M[n + 5, nk + j]     = -1. / dknots[j]
                M[n + 5, nk + j + 1] = 1. / dknots[j]

                M[n + 6, j]          = -9.
                M[n + 6, j + 1]      = 9.
                M[n + 6, nk + j]     = -1. / dknots[j]
                M[n + 6, nk + j + 1] = -2. / dknots[j]

                M[n + 7, j]          = -9.
                M[n + 7, j + 1]      = 9.
                M[n + 7, nk + j]     = -2. / dknots[j]
                M[n + 7, nk + j + 1] = -1. / dknots[j]
            end
            n += 7
        end
    end

    return  vcat(Mineq, M), vcat(rhsineq, zeros(T, size(M, 1)))
end

"""
```
set_left_value(left_value::T, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
```
"""
function set_left_value(left_value::T, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
    M_add = zeros(T, 1, nc)
    M_add[1] = 1.
    return vcat(M, M_add), vcat(rhseq, left_value)
end

"""
```
set_right_value(right_value::T, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
```
"""
function set_right_value(right_value::T, nk::Int, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
    M_add = zeros(T, 1, nc)
    M_add[nk] = 1.
    return vcat(M, M_add), vcat(rhseq, right_value)
end

"""
```
set_min_value(min_value::T, nk::Int, nc::Int, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
    sample_points::AbstractVector{T} = [.017037, .066987, .1465, .25, .37059,
                                        .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}
```

creates the equations enforcing a maximum value for the spline.
"""
function set_min_value(min_value::T, nk::Int, nc::Int, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                       sample_points::AbstractVector{T} = [.017037, .066987, .1465, .25, .37059,
                                                           .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}

    # The default sample points are Chebyshev nodes
    nsamp = length(sample_points)
    ntot = nk + (nk - 1) * nsamp
    Mmin = zeros(T, ntot, nc)
    Mmax = zeros(T, size(Mmin))

    # Constrain values at knots
    for i in 1:nk # This is the same as Mmax[1:nk, 1:nk] = Matrix{T}(I, nk, nk)
        Mmax[i, i] = 1.
    end

    # Intermediate sample points
    t² = tm .^ 2
    t³ = tm .^ 3
    s² = (1. .- tm) .^ 2
    s³ = (1. .- tm) .^ 3
    vals = [3. .* s² .- 2. .* s³,
            3. .* t² .- 2. .* t³,
            -s³ + s², t³ .- t²]

    for j in 1:(nk - 1)
        Mmin[1:nsamp + (j - 1) * nsamp + nk, j .+ [0, 1, nk, nk + 1]] =
            -vals * Diagonal([1, 1, dknots[j], dknots[j]])
    end

    return vcat(Mineq, Mmin), vcat(rhsineq, fill(min_value, ntot))
end

"""
```
set_max_value(max_value::T, nk::Int, nc::Int, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                     sample_points::AbstractVector{T} = [.017037, .066987, .1465, .25, .37059,
                                                         .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}
```

creates the equations enforcing a maximum value for the spline.
"""
function set_max_value(max_value::T, nk::Int, nc::Int, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                       sample_points::AbstractVector{T} = [.017037, .066987, .1465, .25, .37059,
                                                           .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}

    # The default sample points are Chebyshev nodes
    nsamp = length(sample_points)
    ntot = nk + (nk - 1) * nsamp
    Mmax = zeros(T, size(Mmin))

    # Constrain values at knots
    for i in 1:nk # This is the same as Mmax[1:nk, 1:nk] = Matrix{T}(I, nk, nk)
        Mmax[i, i] = 1.
    end

    # Intermediate sample points
    t² = tm .^ 2
    t³ = tm .^ 3
    s² = (1. .- tm) .^ 2
    s³ = (1. .- tm) .^ 3
    vals = [3. .* s² .- 2. .* s³,
            3. .* t² .- 2. .* t³,
            -s³ + s², t³ .- t²]

    # Create equations
    for j in 1:(nk - 1)
        Mmax[1:nsamp + (j - 1) * nsamp + nk, j .+ [0, 1, nk, nk + 1]] =
            vals * Diagonal([1, 1, dknots[j], dknots[j]])
    end

    return vcat(Mineq, Mmax), vcat(rhsineq, fill(max_value, ntot))
end

"""
```
function concave_up_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Tuple{T, T}}}}) where {T <: Real}
```

adds the properties of an everywhere concave up spline.
"""
function concave_up_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}}) where {T <: Real}
    push!(monotone_settings, (concave_up = true, range = Vector{T}(undef, 0)))
end

"""
```
function concave_up_intervals_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                    concave_up_intervals::AbstractMatrix{T}) where {T <: Real}
```

adds information about interval(s) over which the spline should be concave up.
"""
function concave_up_intervals_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                    concave_up_intervals::AbstractMatrix{T}) where {T <: Real}
    for i in 1:size(concave_up_intervals, 1)
        push!(monotone_settings, (concave_up = true, range = sort(concave_up_intervals[i, :])))
    end
end

"""
```
function concave_down_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Tuple{T, T}}}}) where {T <: Real}
```

adds the properties of an everywhere concave down spline.
"""
function concave_up_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}}) where {T <: Real}
    push!(monotone_settings, (concave_up = false, range = Vector{T}(undef, 0)))
end

"""
```
function concave_down_intervals_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                      concave_up_intervals::AbstractMatrix{T}) where {T <: Real}
```

adds information about interval(s) over which the spline should be concave up.
"""
function concave_down_intervals_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                      concave_down_intervals::AbstractMatrix{T}) where {T <: Real}
    for i in 1:size(concave_down_intervals, 1)
        push!(monotone_settings, (concave_up = false, range = sort(concave_down_intervals[i, :])))
    end
end

"""
```
```
"""
function construct_curvature_matrix(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                    nc::Int, nk::Int, dknots::AbstractVector{T}, total_intervals::Int,
                                    Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T}) where {T <: Real}
    L = length(curvature_settings)
    M = zeros(T, 0, nc)
    n = 0

    for i in 1:L
        if isempty(curvature_settings[i][:range])
            # Enter domain specified to be curved in one direction
            if curvature_settings[i][:concave_up]
                for j in 1:(nk - 1)
                    n += 1
                    M[n, j]     = 6. / dknots[j]^2
                    M[n, j + 1] = -6. / dknots[j]^2

                    M[n, nk + j]     = 4. / dknots[j]
                    M[n, nk + j + 1] = 2. / dknots[j]
                end
            else
                for j in 1:(nk - 1)
                    n += 1
                    M[n, j]     = -6. / dknots[j]^2
                    M[n, j + 1] = 6. / dknots[j]^2

                    M[n, nk + j]     = -4. / dknots[j]
                    M[n, nk + j + 1] = -2. / dknots[j]
                end
            end

            n += 1
            if curvature_settings[i][:concave_up]
                M[n, nk - 1] = -6 / dknots[end] ^ 2
                M[n, nk]     = 6 / dknots[end] ^ 2

                M[n, 2 * nk - 1] = -2 / dknots[end]
                M[n, 2 * nk]     = -4 / dknots[end]
            else
                M[n, nk - 1] = 6 / dknots[end] ^ 2
                M[n, nk]     = -6 / dknots[end] ^ 2

                M[n, 2 * nk - 1] = 2 / dknots[end]
                M[n, 2 * nk]     = 4 / dknots[end]
            end
        else
            # Only enforce curvature between the given range limits, knot by knot first
            # and then at the endpoints of the range
        end
    end

    return  vcat(Mineq, M), vcat(rhsineq, zeros(T, size(M, 1)))
end
