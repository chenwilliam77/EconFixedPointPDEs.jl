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
    ŷ, y_scale, y_shift = scale_problem!(x, y, kwargs)

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

    # SCALING ON -> SHIFT/SCALE COEFFICIENTS
    if kwargs[:scaling]
        coef = get_coef(slm)
        if isa(coef, AbstractMatrix)
            coef[:, 1] .-= y_shift
            coef[:, 1] ./= y_scale
            coef[:, 2] ./= y_scale
        else
            coef .-= y_shift
            coef ./= y_scale
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
# p_SLM = slmengine( w,  p_sol,  'concavedown', 'on', 'leftvalue',
#                    p0_norun, 'rightvalue',  p1_norun  , 'knots',   floor( numel(p_sol) /4)   );



end

function SLM_cubic(x::AbstractVector{T}, y::AbstractVector{T}, y_scale, y_shift; knots::Int = 6) where {T <: Real}

    nₓ = length(x)

    # Choose knots
    knot_vals = choose_knots(knots, x)
    dknots = diff(knot_vals)
    if any(dknots .== 0.)
        error("Knots must be distinct.")
    end

    ### Calculate coefficients

    ## Set up
    nc = 2 * knots
    Mineq = zeros(0, nc)
    rhsineq = Vector{T}(undef, 0)
    Meq = zeros(0, nc)
    rhseq = Vector{T}(undef, 0)

    ## Build design matrix

    # Bin data so that xbin has nₓ and xbin specifies into which bin each x value falls
    xbin = bin_sort(x, knots)

    # design matrix
    t  = (x - knot_vals[xbin]) ./ dknots[xbin]
    t² = t.^2
    t³ = t.^3
    s² = (1. .- t).^2
    s³ = (1. .- t).^3

    vals = [3. .* s² .- 2. .* s³;
            3. .* t² .- 2. .* t³;
            (s² - s³) .* dknots[xbin];
            (t³ - t²) .* dknots[xbin]]

    # Coefficients are stored in two blocks,
    # first knots function values, then knots derivatives
    Mdes = accumarray(hcat(repmat(1:nₓ, 4, 1),
                           [xbin; xbin .+ 1.; knots .+ xbin; (knots + 1.) .+ xbin]),
                      vals, sz = (nₓ, nc))
    rhs = y


    # For each of these sections, can we write these as separate functions that go inside a file called
    # property.jl, which holds functions that generate desired output?

    ## Regularizer

    ## C2 continuity across knots

    ## Increasing, either monotone or nx2 array where each row is an interval over which we have increasing

    ## Decreasing

    ## Left and right value

    ## ConcaveUp and ConcaveDown
# cuR = prescription.ConcaveUp;
# L=0;
# if ischar(cuR)
#   if strcmp(cuR,'on')
#     L=L+1;
#     curv(L).knotlist = 'all';
#     curv(L).direction = 1;
#     curv(L).range = [];
#   end
# elseif ~isempty(cuR)
#   for i=1:size(cuR,1)
#     L=L+1;
#     curv(L).knotlist = []; %#ok
#     curv(L).direction = 1; %#ok
#     curv(L).range = sort(cuR(i,:)); %#ok
#   end
# end
# % negative curvature regions
# cdR = prescription.ConcaveDown;
# if ischar(cdR)
#   if strcmp(cdR,'on')
#     L=L+1;
#     curv(L).knotlist = 'all';
#     curv(L).direction = -1;
#     curv(L).range = [];
#   end
# elseif ~isempty(cdR)
#   for i=1:size(cdR,1)
#     L=L+1;
#     curv(L).knotlist = [];
#     curv(L).direction = -1;
#     curv(L).range = sort(cdR(i,:));
#   end
# end
# if L>0
#   % there were at least some regions with specified curvature
#   M = zeros(0,nc);
#   n = 0;
#   for i=1:L
#     if isempty(curv(L).range)
#       % the entire domain was specified to be
#       % curved in some direction
#       for j=1:(nk-1)
#         n=n+1;
#         M(n,j+[0 1]) = curv(i).direction*[6 -6]/(dx(j).^2);
#         M(n,nk+j+[0 1]) = curv(i).direction*[4 2]/dx(j);
#       end
#       n=n+1;
#       M(n,nk+[-1 0]) = curv(i).direction*[-6 6]/(dx(end).^2);
#       M(n,2*nk+[-1 0]) = curv(i).direction*[-2 -4]/dx(end);
#     else
#       % only enforce curvature between the given range limits
#       % do each knot first.
#       for j=1:(nk-1)
#         if (knots(j)<curv(i).range(2)) && ...
#             (knots(j)>=curv(i).range(1))

#           n=n+1;
#           M(n,j+[0 1]) = curv(i).direction*[6 -6]/(dx(j).^2);
#           M(n,nk+j+[0 1]) = curv(i).direction*[4 2]/dx(j);
#         end
#       end

#       % also constrain at the endpoints of the range
#       curv(i).range = max(min(curv(i).range(:),knots(end)),knots(1));
#       [junk,ind] = histc(curv(i).range,knots); %#ok
#       ind(ind==(nk))=nk-1;

#       t = (curv(i).range - knots(ind))./dx(ind);
#       s = 1-t;

#       for j = 1:numel(ind)
#         M(n+j,ind(j)+[0 1 nk nk+1]) = -curv(i).direction* ...
#           [(6 - 12*s(j))./(dx(ind(j)).^2), (6 - 12*t(j))./(dx(ind(j)).^2) , ...
#           (2 - 6*s(j))./dx(ind(j)), (6*t(j) - 2)./dx(ind(j))];
#       end

#       n = n + numel(ind);
#     end
#   end

#   Mineq = [Mineq;M];
#   rhsineq = [rhsineq;zeros(size(M,1),1)];
# end


    # Some more regularization

    # Unpack coefficients into the result structure

    # calculate statistics by translating modelstatistics


    # MAKE SURE TO ADD YSCALE AND YSHIFT TO STATS
end
