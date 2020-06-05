# In case more abstraction is needed
abstract type AbstractSLM{T} end

# TEST THIS CODE BY FITTING MULTIPLE DIFFERENT CURVES FROM LI 2020
# ADD TESTS FOR DECREASING, CONCAVE UP, CONCAVE DOWN,
# AND ALSO ADD TESTS FOR INCREASING DECREASING REGIONS
# DO THIS BY WRITING TESTS IN MATLAB AND THEN SAVING OUTPUT FOR JULIA
# MAKE SURE TO SAVE THE MATLAB SCRIPTS TOO
"""
```
SLM
```

is a port of the main function (with the same name) from
Shape Language Modeling (SLM) by John D'Errico,
who implements least squares spline modeling for curve fitting.

See https://www.mathworks.com/matlabcentral/fileexchange/24443-slm-shape-language-modeling for details about the SLM toolbox.

Note that only the monotonocity features have been ported for now.
Other features, such as convexity and concavity restrictions, may
be added on as-needed basis.

Statistics are computed only if the keyword calculate_stats is true.
"""
mutable struct SLM{T} <: AbstractSLM{T}
    stats::NamedTuple
    x::AbstractVector{T}
    y::AbstractVector{T}
    coef::AbstractArray{T}
end

function Base.show(io::IO, slm::AbstractSLM{T}) where {T <: Real}
    @printf io "SLM with element type %s" string(T)
    @printf io "degree: %i" get_stats(slm)[:degree]
    @printf io "knots:  %i" length(get_stats(slm)[:knots])
end

# Access functions
get_stats(slm::AbstractSLM) = slm.stats
get_x(slm::AbstractSLM) = slm.x
get_y(slm::AbstractSLM) = slm.y
get_coef(slm::AbstractSLM) = slm.coef
eltype(slm::AbstractSLM) = slm.stats

function getindex(slm::AbstractSLM, x::Symbol)
    if x == :stats
        get_stats(slm)
    elseif x == :x
        get_x(slm)
    elseif x == :y
        get_y(slm)
    else
        error("type " * typeof(slm) * " has no field " * string(x))
    end
end

# Main user interface for constructing an SLM object
function SLM(x::AbstractVector{T}, y::AbstractVector{T}; calculate_stats::Bool = false,
             verbose::Symbol = :low, kwargs...) where {T <: Real}

    @assert length(x) == length(y) "x and y must be the same size"
    if verbose == :high
        calculate_stats = true # Statistics will be calculated if verbose is high
    end
    kwargs = Dict(kwargs)

    # Remove nans
    to_remove = isnan.(x) .| isnan.(y)
    if any(to_remove)
        x = x[to_remove]
        y = y[to_remove]

        if haskey(kwargs, :weights)
            error("Weights are not implemented currently.")
            kwargs[:weights] = kwargs[:weights][to_remove]
        end
    end

    # Additional checks
    if haskey(kwargs, :weights)
        @assert length(kwargs[:weights]) == length(x)
    end

    # Add default keyword arguments
    default_slm_kwargs!(kwargs)

    # Scale y. This updates the kwargs
    ŷ = scale_problem!(x, y, kwargs)

    # Determine appropriate fit type
    slm = if kwargs[:degree] == 0
        error("degree 0 has not been implemented")
    elseif kwargs[:degree] == 1
        error("degree 1 has not been implemented")
    elseif kwargs[:degree] == 3

        return SLM_cubic(x, ŷ, kwargs...)
    else
        error("degree $(kwargs[:degree]) has not been implemented")
    end

    # Scaling on -> shift/scale coefficients back
    if kwargs[:scaling]
        coef = get_coef(slm)
        if isa(coef, AbstractMatrix)
            coef[:, 1] .-= kwargs[:y_shift]
            coef[:, 1] ./= kwargs[:y_scale]
            coef[:, 2] ./= kwargs[:y_scale]
        else
            coef .-= kwargs[:y_shift]
            coef ./= kwargs[:y_scale]
        end
    end

    if verbose == :high
        error()
        @info "Model Statistics Report"
        println("Number of data points:      $(length(y))")
        println("Scale factor applied to y   $(y_scale)")
        println("Shift applied to y          $(y_shift)")
        println("Total degrees of freedom:   $(get_stats(slm)[:total_df])")
        println("Net degrees of freedom:     $(get_stats(slm)[:net_df])")
        println("R-squared:                  $(get_stats(slm)[:R2])")
        println("Adjusted R-squared:         $(get_stats(slm)[:R2_adj])")
        println("RMSE:                       $(get_stats(slm)[:RMSE])")
        println("Range of prediction errors: $(get_stats(slm)[:error_range])")
        println("Error quartiles (25%, 75%): $(get_stats(slm)[:quartiles])")
    end
end

function SLM_cubic(x::AbstractVector{T}, y::AbstractVector{T}, y_scale, y_shift;
                   nk::Int = 6, C2::Bool = true, increasing::Bool = false, decreasing::Bool = false,
                   increasing_intervals::AbstractMatrix{T} = Matrix{T}(undef, 0, 0),
                   decreasing_intervals::AbstractMatrix{T} = Matrix{T}(undef, 0, 0),
                   concave_up::Bool = false, concave_down::Bool = false,
                   concave_up_intervals::AbstractMatrix{T} = Matrix{T}(undef, 0, 0),
                   concave_down_intervals::AbstractMatrix{T} = Matrix{T}(undef, 0, 0),
                   left_value::T = NaN, right_value::T = NaN,
                   min_value::T = NaN, max_value::T = NaN,
                   min_max_sample_points::AbstractVector{T} = [.017037, .066987, .1465, .25, .37059,
                                                               .5, .62941, .75, .85355, .93301, .98296]) where {T <: Real}

    nₓ = length(x)

    # Choose knots
    knots = choose_knots(nk, x)
    dknots = diff(knots)
    if any(dknots .== 0.)
        error("Knots must be distinct.")
    end

    ### Calculate coefficients

    ## Set up
    nc = 2 * nk
    Mineq = zeros(0, nc)
    rhsineq = Vector{T}(undef, 0)
    Meq = zeros(0, nc)
    rhseq = Vector{T}(undef, 0)

    ## Build design matrix

    # Bin data so that xbin has nₓ and xbin specifies into which bin each x value falls
    xbin = bin_sort(x, nk)

    # design matrix
    Mdes = construct_design_matrix(x, knots, dknots, xbin)
    rhs = y

    # For each of these sections, can we write these as separate functions that go inside a file called
    # property.jl, which holds functions that generate desired output?

    ## Regularizer
    Mreg = regularizer(dknots, nk)
    rhsreg = zeros(T, nk)

    ## C2 continuity across knots
    if C2
        MC2 = C2_matrix(nk, nc, dknots)
        Meq = vcat(Meq, MC2)
        push!(rhseq, zeros(nk - 2))
    end

    ## Monotonicity restrictions
    @assert !(increasing && decreasing) "Only one of increasing and decreasing can be true"

    monotone_settings = Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}}(undef, 0)
    total_monotone_intervals = 0

    # Monotone increasing
    if increasing
        @assert isempty(increasing_intervals) && isempty(decreasing_intervals) "The spline cannot be monotone increasing and " *
            "have nonempty entries for the keywords increasing_intervals and/or decreasing_intervals"
        total_monotone_intervals += monotone_increasing!(monotone_settings, nk)
    end

    # Increasing intervals: nx2 array where each row is an interval over which we have increasing
    if !isempty(increasing_intervals)
        total_monotone_intervals += increasing_intervals_info!(monotone_settings, knots, increasing_intervals, nk)
    end

    # Monotone decreasing
    if decreasing
        @assert isempty(increasing_intervals) && isempty(decreasing_intervals) "The spline cannot be monotone decreasing and " *
            "have nonempty entries for the keywords increasing_intervals and/or decreasing_intervals"
        monotone_decreasing!(monotone_settings)
    end

    # Decreasing intervals
    if !isempty(decreasing_intervals)
        decreasing_intervals_info!(monotone_settings, increasing_intervals)
    end

    # Add inequalities enforcing monotonicity
    if !isempty(monotone_settings)
        Mineq, rhsineq = construct_monotoncity_matrix(monotone_settings, nc, nk, dknots, total_monotone_intervals, Mineq, rhsineq)
    end

    ## Left and right values
    if !isnan(left_value)
        Meq, rhseq = set_left_value(left_value, Meq, rhseq)
    end

    if !isnan(right_value)
        Meq, rhseq = set_right_value(right_value, nk, Meq, rhseq)
    end

    # Global minimum and maximum values
    if !isnan(min_value)
        Mineq, rhsineq = set_min_value(min_value, nk, nc, Mineq, rhsineq; sample_points = min_max_sample_points)
    end

    if !isnan(max_value)
        Mineq, rhsineq = set_max_value(max_value, nk, nc, Mineq, rhsineq; sample_points = min_max_sample_points)
    end

    ## WE WILL USE NLPMODELS.LinearLeastSquares

    @assert !(concave_up && concave_down) "Only one of concave_up and concave_down can be true"

    curvature_settings = Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, }}}(undef, 0)

    ## Concavity
    if concave_up
        @assert isempty(concave_up_intervals) && isempty(concave_down_intervals) "The spline cannot be concave up and " *
            "have nonempty entries for the keywords concave_up_intervals and/or concave_down_intervals"
        concave_up_info!(curvature_settings)
    end

    if concave_down
        @assert isempty(concave_up_intervals) && isempty(concave_down_intervals) "The spline cannot be concave down and " *
            "have nonempty entries for the keywords concave_up_intervals and/or concave_down_intervals"
        concave_down_info!(curvature_settings)
    end

    if concave_up_intervals
        concave_up_intervals_info!(curvature_settings, concave_up_intervals)
    end

    if concave_down_intervals
        concave_down_intervals_info!(curvature_settings, concave_down_intervals)
    end

    # Add inequalities enforcing monotonicity
    if !isempty(curvature_settings)
        Mineq, rhsineq = constuct_curvature_matrix(curvature_settings, nc, nk, dknots, Mineq, rhsineq)
    end

    # Some more regularization

    # Unpack coefficients into the result structure

    # calculate statistics by translating modelstatistics

    # MAKE SURE TO ADD YSCALE AND YSHIFT TO STATS
    # (; Dict(:a => 5, :b => 6, :c => 7)...) Dict -> NamedTuple
end

"""
```
scale_problem(x::AbstractVector, y::AbstractVector, kwargs)
```

scales y to minimize numerical issues when constructing splines.
"""
function scale_problem(x::AbstractVector, y::AbstractVector, kwargs::Dict{Symbol, Any})
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
        ymin = min(y)
        ymax = max(y)

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
    dknots::AbstractVector{T}, xbin::AbstractVector{Int}) where {T <: Real}
```

constructs the design matrix used to create an SLM.
"""
function construct_design_matrix(x::AbstractVector{T}, knots::AbstractVector{T},
                                 dknots::AbstractVector{T}, xbin::AbstractVector{Int}) where {T <: Real}

    # Create valus used in the design matrix
    t  = (x - knots[xbin]) ./ dknots[xbin]
    t² = t.^2
    t³ = t.^3
    s² = (1. .- t).^2
    s³ = (1. .- t).^3

    vals = [3. .* s² .- 2. .* s³;
            3. .* t² .- 2. .* t³;
            (s² - s³) .* dknots[xbin];
            (t³ - t²) .* dknots[xbin]]

    # Coefficients are stored in two blocks,
    # first nk function values, then nk derivatives
    Mdes = accumarray(hcat(repmat(1:nₓ, 4, 1),
                           [xbin; xbin .+ 1.; nk .+ xbin; (nk + 1.) .+ xbin]),
                      vals, sz = (nₓ, nc))
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
        Mreg[i, i - 1] = dx[i - 1] / 6.
        Mreg[i, i] = (dx[i - 1] + dx[i]) / 3.
        Mreg[i, i + 1] = dx[i] / 6.
    end
    Mreg = cholesky(Mreg).upper # Matrix square root, cholesky is easy to do this

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
    sf[nk, nk] = -6. / dknots[end]^
    sd[nk, nk - 1] = 2 / dknots[end]
    sd[nk, nk] = 4 / dknots[end]
    Mreg = Mreg * hcat(sf, sd)

    # Scale the regularizer before applied to the regularizing parameter
    Mreg ./= norm(Mreg, 1)

    return Mreg
end
