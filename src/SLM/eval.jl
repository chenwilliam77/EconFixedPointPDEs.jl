"""
```
eval(slm::SLM{T}, x::AbstractVector{T}, evalmode::Int = 0) where {T <: Real}
```

evaluates the spline prescribed by `slm` at the values `x`. The optional input
`evalmode` specifies the evaluation of the spline according to

0  -> spline's value at each point in `x`
1  -> first derivative of the spline
2  -> second derivative of the spline
3  -> third derivative of the spline

Note that the inverse of points which fall above the maximum
or below the minimum value of the function will be returned as a NaN.

Currently, no constant extrapolation is supported.
"""
function eval(slm::SLM{T}, x::AbstractVector{T}, evalmode::Int = 0) where {T <: Real}

    ## Set up
    nₓ     = length(x)
    knots  = get_knots(slm)
    nk     = length(knots)
    dknots = diff(knots)
    coef   = get_coef(slm)

    # Extrapolation
    if slm[:extrap] != :none
        error("Extrapolation method $(slm[:extrap]) is not supported.")
    end

    ## Evaluate spline

    # Sort points into bins. Inverse case is handled later -> just zeros for now.
    x_bins = evalmode >= 0 ? bin_sort(x, knots) : zeros(Int, nₓ)

    if get_type(slm) == :cubic
        y = if evalmode == 0
            eval_cubic(x, knots, dknots, coef, x_bins)
        elseif evalmode == 1
            eval_cubic_derivative_1(x, knots, dknots, coef, x_bins)
        elseif evalmode == 2
            eval_cubic_derivative_2(x, knots, dknots, coef, x_bins)
        elseif evalmode == 3
            eval_cubic_derivative_3(x, knots, dknots, coef, x_bins)
        elseif evalmode == -1
            error("The function inverse has not been implemented yet.")
        else
            error("The input `evalmode` cannot be $(evalmode). It must be one of [0, 1, 2, 3].")
        end
    else
        error("Cannot use the spline type $(get_type(slm))")
    end

    return y
end

function eval_cubic(x::AbstractVector{T}, knots::AbstractVector{T}, dknots::AbstractVector{T},
                    coef::AbstractMatrix{T}, x_bins::AbstractVector{Int}) where {T <: Real}

    # Set up
    t  = (x - knots[x_bins]) ./ dknots[x_bins]
    t² = t .^ 2
    t³ = t .^ 3
    s² = (1. .- t) .^ 2
    s³ = (1. .- t) .^ 3

    # No extrapolation yet so directly return
    x_bins_up = x_bins .+ 1

    return (coef[x_bins, 2] .* (s² - s³) +
            coef[x_bins_up, 2] .* (t³ - t²)) .* dknots[x_bins] +
            coef[x_bins, 1] .* (3 .* s² - 2. .* s³) +
            coef[x_bins_up, 1] .* (3. .* t² - 2. .* t³)

end

function eval_cubic_derivative_1(x::AbstractVector{T}, knots::AbstractVector{T}, dknots::AbstractVector{T},
                                 coef::AbstractMatrix{T}, x_bins::AbstractVector{Int}) where {T <: Real}

    # Set up
    t  = (x - knots[x_bins]) ./ dknots[x_bins]
    t² = t .^ 2
    s  = 1. .- t
    s² = (1. .- t) .^ 2

    # No extrapolation yet so directly return
    x_bins_up = x_bins .+ 1

    return coef[x_bins, 2] .* (3. .* s² - 2. .* s) +
        coef[x_bins_up, 2] .* (3. .* t² - 2. .* t) +
        6 .* (coef[x_bins, 1] .* (-s + s²) +
              coef[x_bins_up, 1] .* (t - t²)) ./ dknots[x_bins]
end

function eval_cubic_derivative_2(x::AbstractVector{T}, knots::AbstractVector{T}, dknots::AbstractVector{T},
                                 coef::AbstractMatrix{T}, x_bins::AbstractVector{Int}) where {T <: Real}

    # Set up
    t  = (x - knots[x_bins]) ./ dknots[x_bins]
    s  = 1. .- t

    # No extrapolation yet so directly return
    x_bins_up = x_bins .+ 1

    return (coef[x_bins, 2] .* (2. .- 6. .* s) +
            coef[x_bins_up, 2] .* (6 .* t .- 2.)) ./ dknots[x_bins] +
            (coef[x_bins, 1] .* (6. .- 12. .* s) +
             coef[x_bins_up, 1] .* (6. .- 12. .* t)) ./ (dknots[x_bins] .^ 2)

end

function eval_cubic_derivative_3(x::AbstractVector{T}, knots::AbstractVector{T}, dknots::AbstractVector{T},
                                 coef::AbstractMatrix{T}, x_bins::AbstractVector{Int}) where {T <: Real}

    # No extrapolation yet so directly return
    x_bins_up = x_bins .+ 1

    return 6. .* (coef[x_bins, 2] .+ coef[x_bins_up, 2]) ./ (dknots[x_bins] .^ 2) +
        12. .* (coef[x_bins, 1] .- coef[x_bins_up, 1]) ./ (dknots[x_bins] .^ 3)
end
