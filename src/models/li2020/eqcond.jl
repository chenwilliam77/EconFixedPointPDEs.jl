"""
```
eqcond(m::Li2020)
```

constructs the main loop of the functional iteration when solving
equilibrium in Li (2020) with jumps.
"""
function eqcond(m::Li2020)

    # Unpack items from m that are needed to construct the functional loop
    p_interpolant  = get_setting(m, :p_interpolant)
    xK_interpolant = get_setting(m, :xK_interpolant)
    xg_interpolant = get_setting(m, :xg_interpolant)
    κp_interpolant = get_setting(m, :κp_interpolant)
    firesale_interpolant  = get_setting(m, :firesale_interpolant)
    Q̂_interpolant         = get_setting(m, :Q̂_interpolant)
    Φ = get_setting(m, :Φ)
    Q = get_setting(m, :gov_bond_gdp_level) * get_setting(m, :avg_gdp)
    p₀, p₁ = boundary_conditions(m)[:p]
    max_jump = (p₁ - p₀) / p₁
    κp_grid = get_setting(m, :κp_grid)
    damping_function = get_setting(m, :damping_function)
    f_μK = get_setting(m, :μK)
    xg_tol = get_setting(m, :xg_tol)
    yg_tol = get_setting(m, :yg_tol)
    firesale_bound = get_setting(m, :firesale_bound)
    N_GH = get_setting(m, :N_GH)
    Q̂_tol = get_setting(m, :Q̂_tol)
    Q̂_max_it = get_setting(m, :Q̂_max_it)
    dt = get_setting(m, :dt)
    inside_iter_tol = get_setting(m, :inside_iteration_nlsolve_tol)
    inside_iter_max_iter = get_setting(m, :inside_iteration_nlsolve_max_iter)
    p_tol = get_setting(m, :p_tol)

    # Construct the initial guess for p by interpolating the solution from
    # the no jump equilibrium via SLM

    # The functional loop should return the proposals for the new values of the variables
    # over which we are iterating. For Li (2020), these are p, xg, and Q̂.
    f = function _functional_loop_li2020(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}},
                                         derivs::OrderedDict{Symbol, Vector{S}},
                                         endo::AbstractDict{Symbol, Vector{S}}, θ::NamedTuple;
                                         init_coefs::OrderedDict{Symbol, Vector{S}} =
                                         OrderedDict{Symbol, Vector{S}}(:p => Vector{S}(undef, 0)),
                                         individual_convergence::AbstractDict{Symbol, Vector{S}} =
                                         Dict{Symbol, Vector{S}}(),
                                         verbose::Symbol = :none) where {S <: Real}

        # Unpack variables of interest
        @unpack p, Q̂, xg = funcvar
        ∂p_∂w = derivs[:∂p_∂w]
        @unpack κp, ψ, xK = endo

        # Initialize storages for these values b/c they are separately stored at first due to tempered updating
        p_new  = similar(p)
        xg_new = similar(xg)
        κp_new = κp

        # Add boundaries
        p_new[1]    = p[1]
        p_new[end]  = p[end]
        xg_new[1]   = xg[1]
        xg_new[end] = xg[end]

        ## Step 1: Update for a new round
        p_fitted = extrapolate(interpolate((stategrid[:w], ), p, p_interpolant), Line())
        ψ  .= map((x, y) -> (Φ(x, θ[:χ], θ[:δ]) + θ[:ρ] * (x + y) - θ[:AL]) / (θ[:AH] - θ[:AL]), p, Q̂)
        xK .= ψ ./ stategrid[:w] .* p ./ (p + Q̂)

        solved_κp = trues(length(p)) # always solved at boundaries, but we need to know about the interior
        solved_xK = trues(length(p))
        solved_xg = trues(length(p))
        solved    = trues(length(p))

        ## Step 2: Solve p and xg for each grid point in the interior
        for i in 2:(length(stategrid) - 1)
            wᵢ     = stategrid[:w][i]
            pᵢ     = p[i]
            xKᵢ    = xK[i]
            xgᵢ    = xg[i]
            ∂p_∂wᵢ = ∂p_∂w[i]
            Q̂ᵢ     = Q̂[i]

            κp_new[i], xKᵢ, xg_new[i], succeed_κp, succeed_xK, succeed_xg =
                inside_iteration_li2020(wᵢ, p_fitted, pᵢ, ∂p_∂wᵢ, xKᵢ,
                                        xgᵢ, Q, Q̂ᵢ, max_jump, θ;
                                        κp_grid = κp_grid, xK_interpolant = xK_interpolant,
                                        damping_function = damping_function, xg_tol = xg_tol,
                                        nlsolve_tol = inside_iter_tol,
                                        nlsolve_iter = inside_iter_max_iter, verbose = verbose)

            ψᵢ = xKᵢ * wᵢ / (pᵢ / (pᵢ + Q̂ᵢ))
            p_new[i] = nlsolve(x -> Φ(x[1], θ[:χ], θ[:δ]) .+ θ[:ρ] .* (x[1] .+ Q̂ᵢ) .- (ψᵢ * θ[:AH] + (1. - ψᵢ) * θ[:AL]),
                           [(p₀ + p₁) / 2.]; ftol = p_tol).zero[1]
            solved_κp[i] = succeed_κp
            solved_xK[i] = succeed_xK
            solved_xg[i] = succeed_xg
        end

        # Where was the fixed point system correctly solved?
        solved_index = findall(map(i -> solved_xK[i] & solved_xg[i] & solved_κp[i], 1:(length(stategrid) - 1)))

        # Solve for p over the whole grid by interpolating over points with success
        p_new_slm = SLM(stategrid[:w][solved_index], p_new[solved_index]; increasing = true,
                        concave_down = true, left_value = p₀, right_value = p_new[solved_index[end]],
                        knots = Int(round(length(solved_index) / 2)), init = init_coefs[:p])
        if !isempty(init_coefs[:p]) && length(solved_index) == length(stategrid) - 1 # then we store the coefficients guess for the next round
            init_coefs[:p] .= vec(get_coef(p_new_slm))
        end

        # An alternative to SLM is Dierckx: p_new_spl  = Spline1D(w[solved_index], p_new[solved_index], bc = "extrapolate")
        # but it won't preserve features like monotonicity

        # Evaluate interpolants
        p_new[1:end - 1] .= eval(p_new_slm, stategrid[:w][1:end - 1])
        p_new[end] = p_new[end - 1]
        if !all(solved_xg) # if all(solved_xg), then linear interpolation will just return the same values as they currently are
            # Interpolate other quantities using a 1D cubic Spline
            xg_new_spl = extrapolate(interpolate((stategrid[:w][solved_xg], ), xg_new[solved_xg], xg_interpolant), Line())

            # xg_new_spl = Spline1D(stategrid[:w][solved_xg], xg_new[solved_xg], bc = "extrapolate")

            xg_new[1:end - 1] .= xg_new_spl(stategrid[:w][1:end - 1])
        end

        if !all(solved_κp)
            κp_new_spl = extrapolate(interpolate((stategrid[:w][solved_κp], ), κp_new[solved_κp], κp_interpolant), Line())

            # An alternative: Dierckx
            # κp_new_spl = Spline1D(stategrid[:w][solved_κp], κp_new[solved_κp], bc = "extrapolate")

            κp_new[1:end - 1] .= κp_new_spl(stategrid[:w][1:end - 1])
        end

        if verbose == :high
            println("$(100 * (length(solved_index) + 1) / length(stategrid[:w]))% of the inner iteration is solved")
            println("Calculating Q̂ . . .")
        end

        ## Step 3: Update Q̂ using simulation methods
        calc_Q̂ = haskey(individual_convergence, :Q̂) ? individual_convergence[:Q̂][1] == 0. : true

        if calc_Q̂
            prepare_Q̂!(stategrid, funcvar, derivs, endo, θ, f_μK, Φ, yg_tol, firesale_bound, firesale_interpolant, Q)

            Q̂_new = copy(funcvar[:Q̂])

            Q̂_calculation!(stategrid, Q̂_new, endo[:μw], endo[:σw],
                          endo[:κw], endo[:rf], endo[:rg], endo[:rh], Q, θ[:λ];
                          N_GH = N_GH, tol = Q̂_tol, max_it = Q̂_max_it, Q̂_interp_method = Q̂_interpolant, dt = dt,
                          verbose = verbose)
            # In the original code, a vector of zeros is used as the initial guess
#=            Q̂_new = Q̂_calculation(stategrid, zeros(eltype(m), length(stategrid)), endo[:μw], endo[:σw],
                                  endo[:κw], endo[:rf], endo[:rg], endo[:rh], Q, θ[:λ];
                                  N_GH = N_GH, tol = Q̂_tol, max_it = Q̂_max_it, Q̂_interp_method = Q̂_interpolant, dt = dt,
                                  verbose = verbose)=#

            if !isempty(individual_convergence)
                if sum(abs.(Q̂_new - funcvar[:Q̂])) < individual_convergence[:Q̂][2]
                    individual_convergence[:Q̂][1] = 1.
                    println(verbose, :high, "Q̂ has converged!")
                end
            end
        else
            Q̂_new = funcvar[:Q̂]
        end

        println(verbose, :high, "")

        return p_new, Q̂_new, xg_new # must be ordered in the same order as get_functional_variables(m)
    end
end

"""
```
inside_iteration_li2020(m, stategrid, funcvar, xK, xg, i; verbose = :low)
```

calculates a new xK, xg, and κp within a functional iteration loop
when solving equilibrium in Li (2020) with jumps.
"""
function inside_iteration_li2020(w::S, p_interp::Interpolations.AbstractInterpolation, p::S, ∂p_∂w::S,
                                 xK0::S, xg0::S, Q::S, Q̂::S, max_jump::S,
                                 θ::NamedTuple; κp_grid::AbstractVector{S} = Vector{S}(undef, 0),
                                 xK_max::S = 1e10, κp_residual_error::S = 1e30, xK_residual_error::S = 1e20,
                                 xg_residual_error::S = 1e20,
                                 xK_guess_prop::S = 0.9, xg_guess_prop::S = 0.999, κp_guess::S = 1e-3,
                                 xg_tol::S = 5e-4, residual_tol::Tuple{S, S, S} = (1e-6, 5e-5, 1e-3),
                                 xK_interpolant::Interpolations.InterpolationType = Gridded(Linear()),
                                 damping_function::Function = x -> 3e-8 ./ x,
                                 nlsolve_tol::S = 1e-8, nlsolve_iter::Int = 400, verbose::Symbol = :low) where {S <: Real}

    ## Set up

    # Set flags
    succeed_κp = false
    succeed_xK = false
    succeed_xg = false

    # Create useful functions that will be re-used
    yK_f(xK) = (p ./ (p .+ Q̂) .- w .* xK) ./ (1. .- w)
    yg_f(xg) = (Q ./ (p .+ Q̂) .- w .* xg) ./ (1. .- w)
    yg    = yg_f(xg0) # initializing this value for use in some of the following functions
    σp_f(xK) = ∂p_∂w .* w .* (1. .- w) .* (xK .- yK_f(xK)) ./ (1. .- ∂p_∂w .* w .* (1. .- w) .* (xK .- yK_f(xK))) .* θ[:σK]
    δₓ_f(xK, xg) = max.(θ[:β] .* (xK .+  xg .- 1.) .- xg, 0.)
    I_f(xK, xg) = map(x -> x ? 1. : 0., (θ[:β] .* (xK .+ xg .- 1.) .- xg) .> 0.)
    κd_f(xK, xg) = θ[:θ] .* max.(xK .+ (θ[:ϵ] - 1.), 0.) ./ (xK .+ xg .- 1.)
    κfs_f(xK, xg) = θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg) .* w  ./ (1. .- w)
    κb_f(κp, xK) = θ[:θ] .* (1. .- θ[:ϵ]) .+ (1. .- θ[:θ]) .* (xK .* κp .+ θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg0))
    κh_f(κp, xK) = yK_f(xK) .* κp .+ (1. .- yK_f(xK) .- yg) .* κd_f(xK, xg0) .- κfs_f(xK, xg0)

    ## Step 1. Solve the xK and κp together

    # Define some bounds for where to search
    xK_upper = min(∂p_∂w >= 0. ? (1. + 1. ./ (w .* ∂p_∂w)) : xK_max, p ./ (p .+ Q̂) ./ w)
    xK_lower = 1.
    κp_upper = max_jump
    κp_lower = 0.

    # Define the residual functions for the system of κp and xK
    function κp_residual(F, κp, xK)
        F .= p .- p_interp(w .* (1. .- κb_f(κp, xK)) ./ (1. .- κh_f(κp, xK) .- w .* (κb_f(κp, xK) .- κh_f(κp, xK)))) .-
            κp .+ ((any(κb_f(κp, xK) .> 1.) || any(κh_f(κp, xK) .> 1.) || any(κp .> κp_upper) || any(κp .< κp_lower)) ? κp_residual_error : 0.)
    end
    # xK residual isn't matching up, it looks like somehow one of the 1e20 conditions is getting triggered in Matlab
    function xK_residual(F, κp, xK, xK_res_upper)
        F .= 1e3 .* ((θ[:σK] .+ σp_f(xK)) .^ 2 .* (xK .- yK_f(xK)) .+
                     (θ[:λ] * (1. - θ[:θ])) .* ((κp .+ I_f(xK, xg0) .* (θ[:α] / (1. - θ[:α]) * θ[:β])) ./
                                                (1. .- xK .* κp .- (θ[:α] / (1. - θ[:α])) .* δₓ_f(xK, xg0))) .-
                     θ[:λ] .* (κp .- κd_f(xK, xg0)) ./ (1. .- yK_f(xK) .* κp .- (1. .- yK_f(xK) .- yg) .* κd_f(xK, xg0) .+ κfs_f(xK, xg0)) .-
                     (θ[:AH] - θ[:AL]) ./ p) .+
                     ((any(xK .> xK_res_upper) || any(xK .< xK_lower) || any((θ[:α] / (1. - θ[:α])) .* δₓ_f(xK, xg0) .+ xK .* κp .> 1.)) ?
                     xK_residual_error : 0.)
    end


    # For testing purposes
    if verbose == :high
        upper = (1. - θ[:α] / (1 - θ[:α]) * (θ[:β] * (xg0 - 1) - xg0)) / (0. + θ[:α] / (1 - θ[:α] * θ[:β]))
        xK_test_vec = range(xK_lower, stop = upper, length = 100)
        κp_test_vec = range(κp_lower, stop = .012, length = 200)
        # Plot these residuals xK_residual([0.], 0, xK_test_vec), κp_residual([0.], κp_test_vec, xK)
    end

    # Normal case where bankers are leveraged
    if xK_upper > 1

        # Solve xK for different κp's
        xKs        = similar(κp_grid)
        solved_xKs = BitArray(undef, length(xKs))

        for (j, κp) in enumerate(κp_grid)
            xK_tighter_upper = (1. - θ[:θ] * (1. - θ[:ϵ]) - (1. - θ[:θ]) * (θ[:α] / (1. - θ[:α])) * (θ[:β] * (xg0 - 1.) - xg0)) /
                ((1. - θ[:θ]) * (κp + θ[:α] / (1. - θ[:α]) * θ[:β]))
            xK_guess = xK_guess_prop * min(xK0, (1 - θ[:ϵ]) / κp, xK_tighter_upper)
            if xK_guess < 1.
                xK_guess = (1. + xK0) / 2.
            end
            out = nlsolve((F, xK) -> xK_residual(F, κp, xK, min(xK_tighter_upper, xK_upper)),
                          [xK_guess]; ftol = nlsolve_tol)

            xKs[j] = out.zero[1]
            residual = out.residual_norm # w/1 variable, this is precisely the residual

            if abs(residual) > residual_tol[1] # Is the residual too big?

                # If the solution is on the order of yK = 0, then it's fine
                if abs(xKs[j] - p / (p + Q̂) / w) < residual_tol[2] &&
                    xK_residual([0.], κp, xKs[j], min(xK_upper, xK_tighter_upper))[1] < 0.
                    residual = 0.
                end

                # Full liquidity insurance scenario
                if abs(xKs[j] - (1. - θ[:β]) / θ[:β] + xg0 - 1.) < residual_tol[1]
                    xKs[j] = (1. - θ[:β]) / θ[:β] * xg0 + 1.
                    residual = 0.
                end
            end
            solved_xKs[j] = abs(residual) < residual_tol[1]
        end

        # Now solve for κp
        roots_xK = extrapolate(interpolate((κp_grid[solved_xKs], ), xKs[solved_xKs], xK_interpolant), Line())
        if all(abs.(xKs .+ xg0 .- 1.) .>= residual_tol[1]) # Banks are still leveraged
            out = nlsolve((F, κp) -> κp_residual(F, κp, roots_xK(κp)), [κp_guess]; ftol = nlsolve_tol, iterations = nlsolve_iter)
            κp = out.zero[1]
            succeed_κp = out.f_converged

            xK_tighter_upper = (1. - θ[:θ] * (1. - θ[:ϵ]) - (1. - θ[:θ]) *
                                θ[:α] / (1. - θ[:α]) * (θ[:β] * (xg0 - 1.) - xg0) ) /
                                ((1. - θ[:θ]) * (κp + θ[:α] / (1. - θ[:α]) * θ[:β]))
            out = nlsolve((F, xK) -> xK_residual(F, κp, xK, min(xK_tighter_upper, xK_upper)), [roots_xK(κp)];
                          ftol = nlsolve_tol, iterations = nlsolve_iter)
            xK  = out.zero[1]
            succeed_xK = out.f_converged

            # At the corner xK = 1 / w?
            if abs(xK - p / (p + Q̂) / w) < residual_tol[2]
                xK = p / (p + Q̂) / w
                succeed_xK = xK_residual([0.], κp, xK, min(xK_tighter_upper, xK_upper))[1] < 0
            end

            # Full liquidity insurance scenario
            if abs(xK - (1. - θ[:β]) / θ[:β] * xg0 - 1.) < residual_tol[1]
                xK = (1 - θ[:β]) / θ[:β] * xg0 + 1
                succeed_xK = true
            end

            # Solve for xg now
            xg_upper = Q / (w * (p + Q̂))
            xg_lower = max(θ[:ϵ], 1. / (1. - θ[:β]) * (θ[:β] * (xK - 1.) - (1. - θ[:α]) / θ[:α] * (1. - xK * κp)))
            function xg_residual(F, xg)
                F .= 1e3 .* (I_f(xK, xg) .* (θ[:λ] * (1. - θ[:θ]) * θ[:α] / (1. - θ[:α]) * (1. - θ[:β])) ./
                             (1. .- xK .* κp .- (θ[:α] / (1. - θ[:α])) .* δₓ_f(xK, xg)) .- θ[:λ] .* κd_f(xK, xg) ./
                             (1. .- yK_f(xK) .* κp .- (1. .- yK_f(xK) .- yg_f(xg)) .* κd_f(xK, xg) .+
                              κfs_f(xK, xg)) - damping_function(max.(yg_f(xg), 1e-15))) .+
                              ((any(xg .> xg_upper) || any(xg .< xg_lower)) ? xg_residual_error : 0.)
            end

            if verbose == :high
                xg_test_vec = range(xg_lower, stop = 0.97 * xg_upper, length = 200)
                # Plot xg_residual(xg_test_vec)
            end

            out = nlsolve(xg_residual, [min(xg0, xg_guess_prop * xg_upper)]; ftol = nlsolve_tol, iterations = nlsolve_iter)
            xg  = out.zero[1]
            succeed_xg = out.f_converged

            # Full liquidity insurance
            if abs(xg - θ[:β] / (1. - θ[:β]) * (xK - 1.)) < residual_tol[1]
                xg = θ[:β] / (1. - θ[:β]) * (xK - 1.)
                succeed_xg = true
            end

            # Boundary case
            if (abs(xg - θ[:ϵ]) < residual_tol[1]) && (xg_residual([0.], xg)[1] < 0)
                xg = θ[:ϵ]
                succeed_xg = xg_residual < xg_residual([0.], xg)
            end

            # Second to last case
            if abs(xg - Q / (w * (p + Q̂))) < residual_tol[3]
                xg = Q / (w * (p + Q̂))
                succeed_xg = true
            end

            # Check if the xg solution passes a weaker tolerance
            if !succeed_xg
                out = nlsolve(xg_residual, [min(xg0, xg_guess_prop * xg_upper)]; ftol = xg_tol, iterations = nlsolve_iter)
                xg  = out.zero[1]
                succeed_xg = out.f_converged

                # Full liquidity insurance
                abs(xg - θ[:β] / (1. - θ[:β]) * (xK - 1.))
                if abs(xg - θ[:β] / (1. - θ[:β]) * (xK - 1.)) < residual_tol[2]
                    xg = θ[:β] / (1. - θ[:β]) * (xK - 1.)
                    succeed_xg = true
                end
            end
        end
    end

    ## Step 3: Finish up the inside iteration
    if !(succeed_κp && succeed_xK && succeed_xg)
        warn_str = "At w = $(w), the model has not been solved for "
        to_warn  = Vector{String}(undef, 0)
        if !succeed_κp
            push!(to_warn, "κp")
        end
        if !succeed_xK
            push!(to_warn, "xK")
        end
        if !succeed_xg
            push!(to_warn, "xg")
        end
        all_warns = join(to_warn, ", ")

        if verbose == :low || verbose == :high
            @warn warn_str * all_warns * "."
        end
    end

    return κp, xK, xg, succeed_κp, succeed_xK, succeed_xg
end

"""
```
function prepare_Q̂!(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}}, derivs::OrderedDict{Symbol, Vector{S}},
    endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple, f_μK::Function, Φ::Function, yg_tol::S, firesale_bound::S,
    firesale_interpolant::Function, Q::S) where {S <: Real}
```

calculates the quantities needed for Q̂. Most of this function is a copy of augment_variables!, but
some quantities don't need to be calculated and have been omitted for speed purposes.
"""
function prepare_Q̂!(stategrid::StateGrid, funcvar::OrderedDict{Symbol, Vector{S}}, derivs::OrderedDict{Symbol, Vector{S}},
                    endo::OrderedDict{Symbol, Vector{S}}, θ::NamedTuple, f_μK::Function, Φ::Function, yg_tol::S, firesale_bound::S,
                    firesale_interpolant::Interpolations.InterpolationType, Q::S) where {S <: Real}
    @unpack ψ, xK, yK, yg, σp, σ, σh, σw, μR_rd, rd_rg, rd_rg_H, μb_μh, μw, μp, μK, μR, rd, rg, μb, μh, κp, invst, κb, κd, κh, κfs, firesale_jump, κw, δ_x, indic, rf, rh, rd_rf, μp, μK, μR = endo
    @unpack p, xg, Q̂ = funcvar
    ∂p_∂w = derivs[:∂p_∂w]
    ∂²p∂w² = derivs[:∂²p_∂w²]
    w    = stategrid[:w]

    # Calculate various quantities
    ∂p_∂w  .= differentiate(w, p)    # as Li (2020) does it
    ∂²p∂w² .= differentiate(w, ∂p_∂w)
    invst  .= map(x -> Φ(x, θ[:χ], θ[:δ]), p)
    ψ      .= (invst .+ θ[:ρ] .* (p + Q̂) .- θ[:AL]) ./ (θ[:AH] - θ[:AL])
    ψ[ψ .> 1.] .= 1.

    # Portfolio choices
    xK   .= (ψ ./ w) .* (p ./ (p + Q̂))
    xK[1] = xK[2]
    yK   .= (p ./ (p + Q̂) - w .* xK)  ./ (1. .- w)
    yK[1] = 1.
    yK[end] = yK[end - 1]
    yg   .= (Q ./ (p + Q̂) - w .* xg) ./ (1. .- w)
    yg[end] = yg[end - 1]
    yg[yg .< yg_tol] .= yg_tol # avoid yg < 0
    δ_x   .= max.(θ[:β] .* (xK + xg .- 1.) - xg, 0.)
    indic .= δ_x .> 0.

    # Volatilities
    σp .= ∂p_∂w .* w .* (1. .- w) .* (xK - yK) ./ (1. .- ∂p_∂w .* w .* (1. .- w) .* (xK - yK)) .* θ[:σK]
    σp[end] = 0. # no volatility at the end
    σ  .= xK .* (θ[:σK] .+ σp)
    σh .= yK .* (θ[:σK] .+ σp)
    σw .= (1. .- w) .* (σ - σh)

    # Deal with post jump issues
    firesale_jump .= xK .* κp + (θ[:α] / (1 - θ[:α])) .* δ_x
    firesale_ind  = firesale_jump .< firesale_bound
    firesale_spl  = extrapolate(interpolate((w[firesale_ind], ), firesale_jump[firesale_ind], firesale_interpolant), Line()) # linear extrapolation
    firesale_jump .= firesale_spl(w)

    # Jumps
    κd  .= θ[:θ] .* (xK .+ θ[:ϵ] .- 1.) ./ (xK .+ xg .- 1.) .* (xK .+ θ[:ϵ] .> 1.)
    κb  .= θ[:θ] .* min.(1. - θ[:ϵ], xK) + (1. .- θ[:θ]) .* firesale_jump
    κfs .= (θ[:α] / (1 - θ[:α])) .* δ_x .* w ./ (1. .- w)
    κfs[end] = 0.
    κh  .= yK .* κp + (1. .- yK .- yg) .* κd - κfs
    κw  .= 1. .- (1. .- κb) ./ (1. .- κh .- w .* (κb - κh))

    # Main drifts and interest rates
    μR_rd   .= (θ[:σK] .+ σp) .^ 2 .* xK - θ[:AH] ./ p + (θ[:λ] * (1. - θ[:θ])) .*
        (κp + indic .* (θ[:α] / (1 - θ[:α]) * θ[:β])) ./ (1. .- firesale_jump) +
         ((xK .+ θ[:ϵ]) .< 1.) .* (θ[:λ] * θ[:θ]) ./ (1. .- xK)
    rd_rg   .= (θ[:λ] * (1 - θ[:θ]) * θ[:α] / (1 - θ[:α]) * (1 - θ[:β])) .* indic ./ (1. .- firesale_jump)
    rd_rg_H .= θ[:λ] .* κd ./ (1. .- yK .* κp .- (1. .- yK .- yg) .* κd .+ κfs)
    index_xK = xK .<= 1. # In this scenario, the rd-rg difference must be solved from hh's FOC
    rd_rg[index_xK] = rd_rg_H[index_xK]
    μb_μh   .= (xK - yK) .* μR_rd + (xK .* θ[:AH] - yK .* θ[:AL]) ./ p - (xg - yg) .* rd_rg
    μw      .= (1. .- w) .* (μb_μh + σh .^ 2 - σ .* σh - w .* (σ - σh) .^ 2 -
                       θ[:η] ./ (1. .- w))
    μw[end]  = μw[end - 1] # drift should be similar at the end
    μp      .= ∂p_∂w .* w .* μw + (1. / 2.) .* ∂²p∂w² .* (w .* (1. .- w) .* (σ - σh)) .^ 2
    μp[end]  = μp[end - 1] # drift should be similar at the end
    μK      .= map(x -> f_μK(x, θ[:χ], θ[:δ]), p)
    μR      .= μp .- θ[:δ] + μK + θ[:σK] .* σp - invst ./ p

    # Other interest rates
    rd    .= μR - μR_rd
    rg    .= rd - rd_rg
    rd_rf .= (θ[:λ] * (1. - θ[:θ]) * θ[:α] / (1 - θ[:α]) * (-θ[:β])) .* indic ./ (1. .- firesale_jump)
    rf    .= rd - rd_rf
    rh    .= rd - θ[:λ] .* κd ./ (1. .- yK .* κp - (1. .- yK .- yg) .* κd .+ κfs) # risk free rate for households
end

"""
```
Q̂_calculation!(stategrid::StateGrid, Q̂::AbstractVector{S}, μw::AbstractVector{S}, σw::AbstractVector{S},
    κw::AbstractVector{S}, rf::AbstractVector{S}, rg::AbstractVector{S}, rh::AbstractVector{S}, Q::S, λ::S;
    N_GH::Int = 10, tol::S = 1e-5, max_it::Int = 1000, Q̂_interp_method = Gridded(Linear()),
    testing::Bool = false, verbose::Symbol = :none) where {S <: Real}
```

calculates Q̂ using Gauss-Hermite quadrature. This function modifies Q̂ in place.
"""
function Q̂_calculation!(stategrid::StateGrid, Q̂::AbstractVector{S}, μw::AbstractVector{S}, σw::AbstractVector{S},
                       κw::AbstractVector{S}, rf::AbstractVector{S}, rg::AbstractVector{S}, rh::AbstractVector{S}, Q::S, λ::S;
                       N_GH::Int = 10, tol::S = 1e-5, max_it::Int = 1000,
                       Q̂_interp_method::Interpolations.InterpolationType = Gridded(Linear()), dt::S = 1. / 12.,
                       testing::Bool = false, verbose::Symbol = :low) where {S <: Real}

    ## Set up
    spread = rf - rg
    ϵ_nodes, weight_nodes = gausshermite(N_GH) # approximates exp(-x²)
    ϵ_nodes .*= sqrt(2)      # Normalize ϵ and weight nodes to correctly
    weight_nodes ./= sqrt(π) # approximate a standard Normal distribution

    if testing
        err_vec = Vector{S}(undef, max_it)
    end

    ## Main calculation loop
    err     = 1.
    iter_no = 0
    while (err > tol) && (iter_no < max_it)
        iter_no += 1     # Benchmark tests show that, when length(stategrid[:w]) = 100, copying w/in while loop is faster
        Q̂_last = copy(Q̂) # than having this defined outside the while loop by 5-6 ms. Also 100 times faster than MATLAB!
        Q̂_interp = extrapolate(interpolate((stategrid[:w], ), Q̂_last,
                                           Q̂_interp_method), Line())  # Linear extrapolation
        # Calculate Q̂ everywhere except w = 0
        for i in 2:length(stategrid[:w])

            # Unpack objects characterizing equilibrium
            wᵢ  = stategrid[:w][i]
            μwᵢ = μw[i]
            σwᵢ = σw[i]
            κwᵢ = κw[i]

            # Calculate expectation
            w_noshock   = wᵢ .* (1. .+ μwᵢ * dt .+ σwᵢ .* ϵ_nodes)
            w_shock     = w_noshock .- wᵢ .* κwᵢ
            expectation = (1. - λ * dt) * (weight_nodes' * Q̂_interp(w_noshock)) + λ * dt * (weight_nodes' * Q̂_interp(w_shock))
            Q̂[i]        =  Q * spread[i] * dt + exp(-rf[i] * dt) * expectation
        end

        # Handle case of w = 0
        Q̂[1] = Q * spread[1] / rf[1]

        # Calculate errors
        err = sum(abs.(Q̂ - Q̂_last))
        if testing
            err_vec[iter_no] = err
        end
    end

    if verbose == :high && (iter_no < max_it || err < tol)
        println("Finished calculating Q̂")
    end

    if testing
        return Q̂, err_vec
    else
        return Q̂
    end
end

"""
```
eqcond_nojump(m::Li2020)
```

constructs the ODE characterizing equilibrium
with no jumps and the events deciding termination
"""
function eqcond_nojump(m::Li2020)
    Φ  = get_setting(m, :Φ)
    ∂Φ = get_setting(m, :∂Φ)
    ∂p_∂w0, ∂p_∂wN = get_setting(m, :boundary_conditions)[:∂p_∂w]

    f1 = function _ode_nojump_li2020(p, θ, w)
        if w == 0.
            ∂p_∂w = ∂p_∂w0
        elseif w == 1.
            ∂p_∂w = ∂p_∂wN
        else
            ψ = max(min((Φ(p, θ[:χ], θ[:δ]) + θ[:ρ] * p - θ[:AL]) / (θ[:AH] - θ[:AL]), 1.), 0.)
            xK = (ψ / w)
            yK = (1. - ψ) / (1. - w)
            σp = sqrt((θ[:AH] - θ[:AL]) / (p * (xK - yK))) - θ[:σK]
            σ  = xK * (θ[:σK] + σp)
            σh = yK * (θ[:σK] + σp)

            ∂p_∂w = max(0., σp ./ (w .* (1. - w) .* (xK - yK) .* (θ[:σK] + σp)))
        end

        return ∂p_∂w
    end

    # Terminal condition
    ode_condition(p, w, integrator) = p - get_setting(m, :boundary_conditions)[:p][2]
    ode_affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(ode_condition, ode_affect!)

    return f1, cb
end

# For use during the solve step since we also store the first derivative's boundary conditions
# but they are not used by the ODE solver.
function nojump_ode_init(m::Li2020)
    return get_setting(m, :boundary_conditions)[:p][1]
end
