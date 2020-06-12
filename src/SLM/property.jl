# This file contains functions that create properties for the desired least-square spline

"""
```
default_slm_kwargs!
```

adds the default keyword arguments for an SLM object.
"""
function default_slm_kwargs!(kwargs::Dict, T::Type)

    # degree
    if !haskey(kwargs, :degree)
        kwargs[:degree] = 3
    end

    # # issorted
    # if !haskey(kwargs, :issorted)
    #     kwargs[:issorted] = false
    # end

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
        kwargs[:λ] = convert(T, 1e-4)
    end

    # Monotonicity
    if !haskey(kwargs, :increasing)
        kwargs[:increasing] = false
    end

    if !haskey(kwargs, :decreasing)
        kwargs[:decreasing] = false
    end

    if !haskey(kwargs, :increasing_intervals)
        kwargs[:increasing_intervals] = Matrix{T}(undef, 0, 0)
    end

    if !haskey(kwargs, :decreasing_intervals)
        kwargs[:decreasing_intervals] = Matrix{T}(undef, 0, 0)
    end

    # Curvature
    if !haskey(kwargs, :concave_up)
        kwargs[:concave_up] = false
    end

    if !haskey(kwargs, :concave_down)
        kwargs[:concave_down] = false
    end

    if !haskey(kwargs, :concave_up_intervals)
        kwargs[:concave_up_intervals] = Matrix{T}(undef, 0, 0)
    end

    if !haskey(kwargs, :concave_down_intervals)
        kwargs[:concave_down_intervals] = Matrix{T}(undef, 0, 0)
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
        kwargs[:min_max_sample_points] = convert(Vector{T}, [.017037, .066987, .14645, .25, .37059,
                                                             .5, .62941, .75, .85355, .93301, .98296])
    end

    if !haskey(kwargs, :init)
        kwargs[:init] = Vector{T}(undef, 0)
    end

    return kwargs
end

"""
```
scale_problem!(x::AbstractVector, y::AbstractVector, kwargs)
```

scales y to minimize numerical issues when constructing splines.
"""
function scale_problem!(x::AbstractVector, y::AbstractVector, kwargs::Dict)
    scaling = haskey(kwargs, :scaling) ? kwargs[:scaling] : false
    kwargs[:scaling] = scaling # in case scaling didn't exist already

    if scaling
        # scale y so that the minimum value is 1/phi, and the maximum value phi.
        # where phi is the golden ratio, so phi = (sqrt(5) + 1)/2 = 1.6180...
        # Note that phi - 1/phi = 1, so the new range of yhat is 1. (Note that
        # this interval was carefully chosen to bring y as close to 1 as
        # possible, with an interval length of 1.)
        #
        # The transformation is:
        #   ŷ = y * y_scale + y_shift
        ϕ⁻¹ = (sqrt(5.) - 1.) / 2.;

        # Shift and scale are determined from min and max of y
        ymin = minimum(y)
        ymax = maximum(y)

        y_scale = 1. / (ymax - ymin)

        if isinf(y_scale)
            # If data passed is constant, then no scaling needed
            y_scale = 1.
        end

        y_shift = ϕ⁻¹ - y_scale * ymin

        ŷ = y * y_scale .+ y_shift

        # finally, shift/scale each part of the prescription as necessary.
        # derivative information need only be scaled of course. And since
        # y_scale will always be positive, monotonicity signs
        # and curvature signs will remain unchanged.

        # Those constraints that will be left untouched:
        #  C2, concave_down, concave_up
        #  decreasing, degree, increasing, knots
        #
        # All other constraints will potentially be affected.

        # left_value
        if haskey(kwargs, :left_value)
            kwargs[:left_value] *= y_scale
            kwargs[:left_value] += y_shift
        end

        # right_value
        if haskey(kwargs, :right_value)
            kwargs[:right_value] *= y_scale
            kwargs[:right_value] += y_shift
        end

        # left_value
        if haskey(kwargs, :min_value)
            kwargs[:min_value] *= y_scale
            kwargs[:min_value] += y_shift
        end

        # max_value
        if haskey(kwargs, :max_value)
            kwargs[:max_value] *= y_scale
            kwargs[:max_value] += y_shift
        end
    else
        y_shift = 0.
        y_scale = 1.
        ŷ = copy(y)  # Copy here to avoid accidentally changing original y values
    end

    kwargs[:y_scale] = y_scale
    kwargs[:y_shift] = y_shift

    return ŷ
end

"""
```
construct_design_matrix(x::AbstractVector{T}, knots::AbstractVector{T},
    dknots::AbstractVector{T}, x_bins::AbstractVector{Int}, nₓ::Int, nk::Int, nc::Int) where {T <: Real}
```

constructs the design matrix used to create an SLM.
"""
function construct_design_matrix(x::AbstractVector{T}, knots::AbstractVector{T},
                                 dknots::AbstractVector{T}, x_bins::AbstractVector{Int},
                                 nₓ::Int, nk::Int, nc::Int) where {T <: Real}

    # Create values used in the design matrix
    t  = (x - knots[x_bins]) ./ dknots[x_bins]
    t² = t.^2
    t³ = t.^3
    s² = (1. .- t) .^ 2
    s³ = (1. .- t) .^ 3

    vals = [3. .* s² .- 2. .* s³;
            3. .* t² .- 2. .* t³;
            (s² - s³) .* dknots[x_bins];
            (t³ - t²) .* dknots[x_bins]]

    # Coefficients are stored in two blocks,
    # first nk function values, then nk derivatives
    # Must use vector of CartesianIndex for the subscripts (unlike in Matlab)
    Mdes = accumarray([CartesianIndex(i, j) for (i, j) in
                       zip(repeat(collect(1:nₓ), 4),
                           [x_bins; x_bins .+ 1; nk .+ x_bins; (nk + 1) .+ x_bins])],
                      vals, (nₓ, nc))

    return Mdes
end

"""
```
construct_regularizer(dknots::AbstractVector{T}, nk::Int) where {T <: Real}
```

constructs the regularizer equations used to fit the least-squares spline
when constructing an SLM.
"""
function construct_regularizer(dknots::AbstractVector{T}, nk::Int) where {T <: Real}
    # We are integrating the piecewise linear f''(x)
    # as a quadratic form in terms of the (unknown) second
    # derivatives at the knots
    Mreg = zeros(T, nk, nk)
    Mreg[1, 1] = dknots[1] / 3.
    Mreg[1, 2] = dknots[1] / 6.
    Mreg[nk, nk - 1] = dknots[end] / 6.
    Mreg[nk, nk] = dknots[end] / 3.
    for i in 2:(nk - 1)
        Mreg[i, i - 1] = dknots[i - 1] / 6.
        Mreg[i, i] = (dknots[i - 1] + dknots[i]) / 3.
        Mreg[i, i + 1] = dknots[i] / 6.
    end
    Mreg = cholesky(Mreg).U # Matrix square root, cholesky is easy way to do this

    # Write second derivativeas as a function of
    # the function values and first derivatives
    sf = zeros(T, nk, nk)
    sd = zeros(T, nk, nk)
    for i in 1:(nk - 1)
        sf[i, i] = -(6. / dknots[i]^2)
        sf[i, i + 1] = 6. / dknots[i]^2
        sd[i, i] = -4. / dknots[i]
        sd[i, i + 1] = -2. / dknots[i]
    end
    sf[nk, nk - 1] = 6. / dknots[end]^2
    sf[nk, nk] = -6. / dknots[end]^2
    sd[nk, nk - 1] = 2 / dknots[end]
    sd[nk, nk] = 4. / dknots[end]
    Mreg = Mreg * hcat(sf, sd)

    # Scale the regularizer before applied to the regularizing parameter
    Mreg ./= opnorm(Mreg, 1)

    return Mreg
end

"""
```
C2_matrix(nk::Int, nc::Int, dknots::AbstractVector{T},
    Meq::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
```

constructs the matrix of equations that enforce twice-continuous differentiability.
"""
function C2_matrix(nk::Int, nc::Int, dknots::AbstractVector{T}, Meq::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
    MC2 = zeros(T, nk - 2, nc)
    for i in 1:(nk - 2)
        MC2[i, i] = 6. / dknots[i]^2
        MC2[i, i + 1] = -6. / dknots[i]^2 + 6. / dknots[i + 1]^2
        MC2[i, i + 2] = -6. / dknots[i + 1]^2

        MC2[i, nk + i] = 2. / dknots[i]
        MC2[i, nk + i + 1] = 4. / dknots[i] + 4. / dknots[i + 1]
        MC2[i, nk + i + 2] =  2. / dknots[i]
    end

    return vcat(Meq, MC2), vcat(rhseq, zeros(T, nk - 2))
end

"""
```
function monotone_increasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
    nk::Int) where {T <: Real}
```

adds the properties of a monotone increasing spline.
"""
function monotone_increasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
                              nk::Int)
    push!(monotone_settings, (knotlist = (1, nk), increasing = true))
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
        out += knotlist[2] - knotlist[1]
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
function monotone_decreasing!(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
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
        out += knotlist[2] - knotlist[1]
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
function construct_monotonicity_matrix(monotone_settings::Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}},
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
                M[n + 4, nk + j]     = -1. * dknots[j]
                M[n + 4, nk + j + 1] = 1. * dknots[j]

                M[n + 5, j]          = 3.
                M[n + 5, j + 1]      = -3.
                M[n + 5, nk + j]     = 1. * dknots[j]
                M[n + 5, nk + j + 1] = -1. * dknots[j]

                M[n + 6, j]          = 9.
                M[n + 6, j + 1]      = -9.
                M[n + 6, nk + j]     = 1. * dknots[j]
                M[n + 6, nk + j + 1] = 2. * dknots[j]

                M[n + 7, j]          = 9.
                M[n + 7, j + 1]      = -9.
                M[n + 7, nk + j]     = 2. * dknots[j]
                M[n + 7, nk + j + 1] = 1. * dknots[j]
            else
                M[n + 1, j]          = -1.
                M[n + 1, j + 1]      = 1
                M[n + 2, nk + j]     = 1.
                M[n + 3, nk + j + 1] = 1.

                M[n + 4, j]          = -3.
                M[n + 4, j + 1]      = 3.
                M[n + 4, nk + j]     = 1. * dknots[j]
                M[n + 4, nk + j + 1] = -1. * dknots[j]

                M[n + 5, j]          = -3.
                M[n + 5, j + 1]      = 3.
                M[n + 5, nk + j]     = -1. * dknots[j]
                M[n + 5, nk + j + 1] = 1. * dknots[j]

                M[n + 6, j]          = -9.
                M[n + 6, j + 1]      = 9.
                M[n + 6, nk + j]     = -1. * dknots[j]
                M[n + 6, nk + j + 1] = -2. * dknots[j]

                M[n + 7, j]          = -9.
                M[n + 7, j + 1]      = 9.
                M[n + 7, nk + j]     = -2. * dknots[j]
                M[n + 7, nk + j + 1] = -1. * dknots[j]
            end
            n += 7
        end
    end

    return  vcat(Mineq, M), vcat(rhsineq, zeros(T, size(M, 1)))
end

"""
```
set_left_value(left_value::T, nc:Int, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
```
"""
function set_left_value(left_value::T, nc::Int, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
    M_add = zeros(T, 1, nc)
    M_add[1] = 1.
    return vcat(M, M_add), vcat(rhseq, left_value)
end

"""
```
set_right_value(right_value::T, nc::Int, nk::Int, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
```
"""
function set_right_value(right_value::T, nc::Int, nk::Int, M::AbstractMatrix{T}, rhseq::AbstractVector{T}) where {T <: Real}
    M_add = zeros(T, 1, nc)
    M_add[nk] = 1.
    return vcat(M, M_add), vcat(rhseq, right_value)
end

"""
```
set_min_value(min_value::T, nk::Int, nc::Int, dknots::AbstractVector{T}, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
    sample_points::AbstractVector{T} = [.017037, .066987, .14645, .25, .37059,
                                        .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}
```

creates the equations enforcing a maximum value for the spline.
"""
function set_min_value(min_value::T, nk::Int, nc::Int, dknots::AbstractVector{T}, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                       sample_points::AbstractVector{T} = [.017037, .066987, .14645, .25, .37059,
                                                           .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}

    # The default sample points are Chebyshev nodes
    nsamp = length(sample_points)
    ntot = nk + (nk - 1) * nsamp
    Mmin = zeros(T, ntot, nc)

    # Constrain values at knots
    for i in 1:nk # This is the same as Min[1:nk, 1:nk] = -Matrix{T}(I, nk, nk)
        Mmin[i, i] = -1.
    end

    # Intermediate sample points
    t² = sample_points .^ 2
    t³ = sample_points .^ 3
    s² = (1. .- sample_points) .^ 2
    s³ = (1. .- sample_points) .^ 3
    vals = hcat(3. .* s² .- 2. .* s³,
            3. .* t² .- 2. .* t³,
             s² - s³, t³ .- t²)

    for j in 1:(nk - 1)
        Mmin[(1:nsamp) .+ ((j - 1) * nsamp + nk), j .+ [0, 1, nk, nk + 1]] =
            -vals * Diagonal([1., 1., dknots[j], dknots[j]])
    end

    return vcat(Mineq, Mmin), vcat(rhsineq, fill(-min_value, ntot))
end

"""
```
set_max_value(max_value::T, nk::Int, nc::Int, dknots::AbstractVector{T}, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                     sample_points::AbstractVector{T} = [.017037, .066987, .14645, .25, .37059,
                                                         .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}
```

creates the equations enforcing a maximum value for the spline.
"""
function set_max_value(max_value::T, nk::Int, nc::Int, dknots::AbstractVector{T}, Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T};
                       sample_points::AbstractVector{T} = [.017037, .066987, .14645, .25, .37059,
                                                           .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}

    # The default sample points are Chebyshev nodes
    nsamp = length(sample_points)
    ntot = nk + (nk - 1) * nsamp
    Mmax = zeros(T, ntot, nc)

    # Constrain values at knots
    for i in 1:nk # This is the same as Mmax[1:nk, 1:nk] = Matrix{T}(I, nk, nk)
        Mmax[i, i] = 1.
    end

    # Intermediate sample points
    t² = sample_points .^ 2
    t³ = sample_points .^ 3
    s² = (1. .- sample_points) .^ 2
    s³ = (1. .- sample_points) .^ 3
    vals = hcat(3. .* s² .- 2. .* s³,
            3. .* t² .- 2. .* t³,
            -s³ + s², t³ .- t²)

    # Create equations
    for j in 1:(nk - 1)
        Mmax[(1:nsamp) .+ ((j - 1) * nsamp + nk), j .+ [0, 1, nk, nk + 1]] =
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
    push!(curvature_settings, (concave_up = true, range = Vector{T}(undef, 0)))
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
        push!(curvature_settings, (concave_up = true, range = sort(concave_up_intervals[i, :])))
    end
end

"""
```
function concave_down_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Tuple{T, T}}}}) where {T <: Real}
```

adds the properties of an everywhere concave down spline.
"""
function concave_down_info!(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}}) where {T <: Real}
    push!(curvature_settings, (concave_up = false, range = Vector{T}(undef, 0)))
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
        push!(curvature_settings, (concave_up = false, range = sort(concave_down_intervals[i, :])))
    end
end

"""
```
function construct_curvature_matrix(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                    nc::Int, nk::Int, knots::AbstractVector{T}, dknots::AbstractVector{T},
                                    Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T}) where {T <: Real}
```

creates the inequalities enforcing curvature for the spline
"""
function construct_curvature_matrix(curvature_settings::Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{T}}}},
                                    nc::Int, nk::Int, knots::AbstractVector{T}, dknots::AbstractVector{T},
                                    Mineq::AbstractMatrix{T}, rhsineq::AbstractVector{T}) where {T <: Real}

    L = length(curvature_settings)
    if L > 1
        # Then not entirely concave up or down so need to handle construction of M
        dim1 = nk - 1
        for i in 1:L
            end_range = map(a -> max(min(a, knots[end]), knots[1]), curvature_settings[i][:range])
            dim1 += length(bin_sort(end_range, knots))
        end
        M = zeros(T, dim1, nc)
    else
        M = zeros(T, nk, nc)
    end
    n = 0

    for i in 1:L
        if isempty(curvature_settings[i][:range])
            # Enter domain specified to be curved in one direction
            if curvature_settings[i][:concave_up]
                for j in 1:(nk - 1)
                    n += 1
                    M[n, j]     = 6. / dknots[j] ^ 2
                    M[n, j + 1] = -6. / dknots[j] ^ 2

                    M[n, nk + j]     = 4. / dknots[j]
                    M[n, nk + j + 1] = 2. / dknots[j]
                end
            else
                for j in 1:(nk - 1)
                    n += 1
                    M[n, j]     = -6. / dknots[j] ^ 2
                    M[n, j + 1] = 6. / dknots[j] ^ 2

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
            curv_setting = curvature_settings[i]
            if curv_setting[:concave_up]
                for j in 1:(nk - 1)
                    if (knots[j] < curv_setting[:range][2]) &&
                        (knots[j] >= curv_setting[:range][1])

                        n += 1
                        M[n, j]     = 6. / dknots[j] ^ 2
                        M[n, j + 1] = -6. / dknots[j] ^ 2

                        M[n, nk + j]     = 4. / dknots[j]
                        M[n, nk + j + 1] = 2. / dknots[j]
                    end
                end
            else
                for j in 1:(nk - 1)
                    if (knots[j] < curv_setting[:range][2]) &&
                        (knots[j] >= curv_setting[:range][1])

                        n += 1
                        M[n, j]     = -6. / dknots[j] ^ 2
                        M[n, j + 1] = 6. / dknots[j] ^ 2

                        M[n, nk + j]     = -4. / dknots[j]
                        M[n, nk + j + 1] = -2. / dknots[j]
                    end
                end
            end

            end_range = map(a -> max(min(a, knots[end]), knots[1]), curv_setting[:range])
            ind = bin_sort(end_range, knots)
            t = (end_range - knots[ind]) ./ dknots[ind]
            s = 1. .- t

            if curv_setting[:concave_up]
                for j in 1:length(ind)
                    M[n + j, ind[j]]          = -(6. - 12. * s[j]) / dknots[ind[j]] ^ 2
                    M[n + j, ind[j] + 1]      = -(6. - 12. * t[j]) / dknots[ind[j]] ^ 2
                    M[n + j, ind[j] + nk]     = -(2. - 6. * s[j]) / dknots[ind[j]]
                    M[n + j, ind[j] + nk + 1] = -(6. * t[j] - 2.) / dknots[ind[j]]
                end
            else
                for j in 1:length(ind)
                    M[n + j, ind[j]]          = (6. - 12. * s[j]) / dknots[ind[j]] ^ 2
                    M[n + j, ind[j] + 1]      = (6. - 12. * t[j]) / dknots[ind[j]] ^ 2
                    M[n + j, ind[j] + nk]     = (2. - 6. * s[j]) / dknots[ind[j]]
                    M[n + j, ind[j] + nk + 1] = (6. * t[j] - 2.) / dknots[ind[j]]
                end
            end

            n += length(ind)
        end
    end

    return  vcat(Mineq, M), vcat(rhsineq, zeros(T, size(M, 1)))
end
