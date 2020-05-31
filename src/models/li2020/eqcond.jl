"""
```
eqcond(m::Li2020)
```

constructs the main loop of the functional iteration when solving
equilibrium in Li (2020) with jumps.
"""
function eqcond(m::Li2020)
    # θ = parameters_to_named_tuple(get_parameters(m))

    # Unpack items from m that are needed to construct the functional loop
    p_fitted_interpolant = get_setting(m, :p_fitted_interpolant)
    Φ = get_setting(m, :Φ)
    Q = get_setting(m, :gov_bond_gdp_level) * get_setting(m, :avg_gdp)
    p₀, p₁ = boundary_conditions(m)[:p]
    max_jump = (p₁ - p₀) / p₁
    κp_grid = get_setting(m, :κp_grid)
    damping_function = get_setting(m, :damping_function)

    # The functional loop should return the proposals for the new values of the variables
    # over which we are iterating. For Li (2020), these are p, xg, and Q̂.
    f = function _functional_loop_li2020(stategrid::StateGrid, diffvar::OrderedDict{Symbol, AbstractVector},
                                         endo::OrderedDict{Symbol, AbstractVector{S}}, θ::NamedTuple;
                                         verbose::Symbol = :low) where {S <: Real}

        # Unpack variables of interest
        p    = diffvar[:p]
        ∂p∂w = diffvar[:∂p∂w]
        κp   = diffvar[:κp]
        Q̂    = endo[:Q̂]
        xg   = endo[:xg]
        ψ    = endo[:ψ]
        xK   = endo[:xK]

        # Initialize storages for these values b/c they are separately stored at first due to tempered updating
        p_new  = similar(p)
        xg_new = similar(xg)
        κp_new = similar(κp)

        ## Step 1: Update for a new round
        p_fitted = interpolate((stategrid[:w], ), p, p_fitted_interpolant)
        ψ  .= map((x, y) -> (Φ(x, θ[:χ], θ[:δ]) + θ[:ρ] * (x + y) - θ[:AL]) / (θ[:AH] - θ[:AL]), p, Q̂)
        xK .= ψ ./ stategrid[:w] .* p ./ (p + Q̂)

        solved_κp = BitArray(undef, length(p))
        solved_xK = BitArray(undef, length(p))
        solved_xg = BitArray(undef, length(p))
        solved    = BitArray(undef, length(p))

        ## Step 2: Solve p and xg for each grid point
        for (i, wᵢ) in enumerate(stategrid[:w])
            pᵢ    = p[i]
            xKᵢ   = xK[i]
            xgᵢ   = xg[i]
            ∂p∂wᵢ = ∂p∂w[i]
            Q̂ᵢ    = Q̂[i]

            κp_new[i], xKᵢ, xg_new[i], succeed_κp, succeed_xK, succeed_xg =
                inside_iteration(wᵢ, p_fitted, pᵢ, ∂p∂wᵢ, xKᵢ,
                                 xgᵢ, Q, Q̂ᵢ, max_jump, θ;
                                 κp_grid = κp_grid,
                                 damping_function = damping_function,
                                 verbose = verbose)

            ψᵢ = xKᵢ / wᵢ / (pᵢ / (pᵢ + Q̂ᵢ))
            p_new[i] = nlsolve(x -> Φ(x, θ[:χ], θ[:δ]) .+ θ[:ρ] .* (x .+ Q̂ᵢ) .- (ψᵢ * θ[:AH] + (1. - ψᵢ) * θ[:AL]),
                           [(p₀ + p₁) / 2]., autodiff = :forward)
            solved_κp[i] = succeed_κp
            solved_xK[i] = succeed_xK
            solved_xg[i] = succeed_xg
        end

        # Where was the fixed point system correctly solved?
        solved_index = findall(map(i -> solved_xK[i] && solved_xg[i] && solved_κp[i], 1:length(w)))

        if verbose == :low || verbose == :high
            @info "$(100 * (length(solved_index) + 1) / length(w))% is solved"
        end

        # Solve for p over the whole grid by interpolating over points with success
        # TO BE DONE: use least-squares spline w/monotonicity for q
        p_new_spl  = Spline1D(w[solved_index], p_new[solved_index], bc = "extrapolate")
        xg_new_spl = Spline1D(w[solved_xg], xg_new[solved_xg], bc = "extrapolate")
        κp_new_spl = Spline1D(w[solved_κp], κp_new[solved_κp], bc = "extrapolate")
        p_new  .= map(x -> p_new_spl(x), w)
        xg_new .= map(x -> xg_new_spl(x), w)
        κp_new .= map(x -> κp_new_spl(x), w)

        ## Step 3: Update Q̂ using simulation methods



        return p_new, xg_new, Q̂_new
    end
end

"""
```
inside_iteration(m, stategrid, diffvar, xK, xg, i; verbose = :low)
```

calculates a new xK, xg, and κp within a functional iteration loop
when solving equilibrium in Li (2020) with jumps.
"""
function inside_iteration(w::S, p_interp, p::S, ∂p∂w::S, xK0::S, xg0::S, Q::S, Q̂::S, max_jump::S,
                          θ::NamedTuple; κp_grid::AbstractVector{S} = Vector{S}(undef, 0),
                          xK_max::S = 1e10, κp_residual_error::S = 1e30, xK_residual_error::S = 1e20,
                          xg_residual_error::S = 1e20,
                          xK_guess_prop::S = 0.9, xg_guess_prop, κp_guess::S = 1e-3,
                          residual_tol::Tuple{S, S, S} = (1e-6, 1e-5, 1e-3),
                          xK_interpolant::Function = Gridded(Linear()),
                          damping_function::Function = x -> 3e-8 ./ x,
                          verbose::Symbol = :low) where {S <: Real}

    ## Set up

    # Create useful functions that will be re-used
    yK_f  = xK -> (p ./ (p .+ Q̂) .- w .* xK) ./ (1. .- w)
    yg_f  = xg -> (Q ./ (p .+ Q̂) .- w .* xg) ./ (1. .- w)
    yg    = yg_f(xg) # initializing this value for use in some of the following functions
    σp_f  = xK -> ∂p∂w .* w .* (1. .- w) .* (xK .- yK_f(xK)) ./ (1. .- ∂p∂w .* w .* (1. .- w) .* (xK .- yK_f(xK))) .* θ[:σK]
    δₓ_f  = (xK, xg) -> max.(θ[:β] .* (xK .+ xg .- 1.) .- xg, 0.)
    I_f   = (xK, xg) -> θ[:β] .* (xK .+ xg .- 1.) .- xg .> 0.
    κd_f  = (xK, xg) -> θ[:θ] .* max.(xK .+ 1e-1, 0.) ./ (xK .+ xg .- 1.)
    κfs_f = (xK, xg) = θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg) .* w  ./ (1. .- w)
    κb_f  = (κp, xK) -> θ[:θ] .* (1. .- θ[:e]) .+ (1. .- θ[:θ]) .* (xK .* κp .+ θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg))
    κh_f  = (κp, xK) -> yK_f(xK) .* κp .+ (1. .- yK_f(xK) .- yg) .* κd_f(xK, xg) .- κfs_f(xK, xg)

    ## Step 1. Solve the xK and κp together

    # Define some bounds for where to search
    xK_upper = min(∂p∂w >= 0. ? (1. + 1. ./ (w .* ∂p∂w)) : xK_max, p ./ (p .+ Q̂) ./ w)
    xK_lower = 1.
    κp_upper = max_jump
    κp_lower = 0.

    # Define the residual functions for the system of κp and xK
    function κp_residual(F, κp, xK)
        F.= p .- interp(w .* (1. .- κb_f(κp, xK)) ./ (1. .- κh_f(κp, xK) .- w .* (κb_f(κp, xK) .- κh_f(κp, xK))), p_interp) .-
            κp .+ (any(κb_f(κp, xK) .> 1.) || any(κh_f(κp, xK) .> 1.) || any(κp .> κp_upper) || any(κp .< κp_lower)) ? κp_residual_error : 0.
    end
    function xK_residual(F, κp, xK, xK_res_upper)
        F .= 1e3 .* ((θ[:σK] .+ σp_f(xK)).^2 .* (xK .- yK_f(xK)) .+
                     θ[:λ] .* (1. .- θ[:θ]) .* ((κp .+ I_f(xK, xg) .* θ[:α] ./ (1. .- θ[:α]) .* θ[:β]) ./
                                                (1. .- xK .* κp .- θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg))) .-
                     θ[:λ] .* (κp .- κd_f(xK, x)) ./ (1. .- yK_f(xK) .* κp .- (1. .- yK_f(xK) .- yg) .* κd_f(xK, xg) .+ κfs_f(xK, xg)) .-
                     (θ[:AH] .- θ[:AL]) ./ p) .+
                     (any(xK .> xK_res_upper) || any(xK .< xK_lower) || any(θ[:α] ./ (1. .- θ[:α]) .* δₓ_f(xK, xg) .+ xK .* κp .> 1.)) ?
                     xK_residual_error : 0.
    end

    # For testing purposes
    if verbose == :high
        upper = (1. - θ[:α] / (1 - θ[:α]) * (θ[:β] * (xg - 1) - xg)) / (0. + θ[:α] / (1 - θ[:α] * θ[:β]))
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
            xK_tighter_upper = (1. - θ[:θ] * (1. - θ[:e]) - (1. - θ[:θ]) * (θ[:α] / (1. - θ[:α])) * (θ[:β] * (xg - 1.) - xg)) /
                ((1. - θ[:θ]) * (κp + θ[:α] / (1. - θ[:α]) * θ[:β]))
            out = nlsolve((F, xK) -> xK_residual(F, κp, xK, xK_tighter_upper),
                          [xK_guess_prop * min(xK0, (1 - θ[:e]) / κp, xK_tighter_upper)], autodiff = :forward)
            xKs[j] = out.zero
            residual = out.residual_norm # w/1 variable, this is precisely the residual

            if abs(residual) > residual_tol[1] # Is the residual too big?

                # If the solution is on the order of yK = 0, then it's fine
                if abs(xKs[j] - p / (p + Q̂) / w) < residual_tol[2] && xK_residual(κp, xKs[j], xK_tighter_upper) < 0.

                    residual = 0.
                end

                # Full liquidity insurance scenario
                if abs(xKs[j] - (1. - θ[:β]) / θ[:β] + xg - 1.) < residual_tol[1]
                    xKs[j] = (1. - θ[:β]) / θ[:β] * xg + 1.
                    residual = 0.
                end
            end
            solved_xKs[j] = abs(residual) < residual_tol[1]
        end

        # Now solve for κp
        roots_xK = interpolate((κps[solved_xKs], ), xKs[solved_xKs], xK_interpolant)
        if all(abs.(xKs + xg - 1.) .>= residual_tol[1]) # Banks are still leveraged

            out = nlsolve((F, κp) -> κp_residual(F, κp, roots_xK(κp)), [κp_residal_guess], autodiff = :forward)
            κp = out.zero
            succeed_κp = out.f_converged

            xK_tighter_upper = (1. - θ[:θ] * (1. - θ[:e]) - (1. - θ[:θ]) *
                                θ[:α] / (1. - θ[:α]) * (θ[:β] * (xg - 1.) - xg) ) /
                                ((1. - θ[:θ]) * (κp + θ[:α] / (1. - θ[:α]) * θ[:β]))
            out = nlsolve((F, xK) -> xK_residual(F, κp, xK, xK_tighter_upper), [roots_xK(κp)], autodiff = :forward)
            xK  = out.zero
            succeed_xK = out.f_converged

            # At the corner xK = 1 / w?
            if abs(xK - p / (p + Q̂) / w) < residual_tol[1]
                xK = p / (p + Q̂) / w
                succeed_xK = xK_residual(κp, xK) < 0
            end

            # Full liquidity insurance scenario
            if abs(xK - (1. - θ[:β]) / θ[:β] * xg - 1.) < residual_tol[1]
                xK = (1 - θ[:β]) / θ[:β] * xg + 1
                succeed_xK = true
            end

            # Solve for xg now
            xg_upper = Q / (w * (p + Q̂))
            xg_lower = max(θ[:e], 1. / (1. - θ[:β]) * (θ[:β] * (xK - 1.) - (1. - θ[:α]) / θ[:α] * (1. - xK * κp)))
            function xg_residual(F, xg)
                F .= 1e3 .* (I_f(xK, xg) .* θ[:λ] .* (1. .- θ[:θ]) .* (θ[:α] ./ (1. .- θ[:α]) .* (1. .- θ[:β])) ./
                             (1. .- xK .* κp .- θ[:α] ./ (1. .- θ[:α]) * δₓ_f(xK, xg)) .- θ[:λ] .* κd_f(xK, xg) ./
                             (1. .- yK_f(xK) .* κp .- (1. .- yK_f(xK) .- yg_f(xg)) .* κd_f(xK, xg) .+
                              κfs_f(xK, xg)) - damping_function(max.(yg_f(xg), 0.))) .+
                              (any(xg .> xg_upper) || any(xg .< xg_lower)) ? xg_residual_error : 0.
            end

            if verbose == :high
                xg_test_vec = range(xg_lower, stop = 0.97 * xg_upper, length = 200)
                # Plot xg_residual(xg_test_vec)
            end

            out = nlsolve(xg_residual, [min(xg, xg_guess_prop * xg_upper)], autodiff = :forward)
            xg  = out.zero
            succeed_xg = out.f_converged

            # Full liquidity insurance
            if abs(xg - θ[:β] / (1. - θ[:β]) * (xK - 1.)) < residual_tol[1]
                xg = θ[:β] / (1. - θ[:β]) * (xK - 1.)
                succeed_xg = true
            end

            # Boundary case
            if (abs(xg - θ[:e]) < residual_tol[1]) && (xg_residual(F, xg) < 0)
                xg = θ[:e]
                succeed_xg = xg_residual < xg_residual([0.], xg)
            end

            # Last case
            if abs(xg - Q / (w * (p * Q̂))) < residual_tol[3]
                xg = Q / (w * (p * Q̂))
                succeed_xg = true
            end
        end
    else
        succeed_κp = false
        succeed_xK = false
        succeed_xg = false
    end

    ## Step 3: Finish up the inside iteration
    if !(succeed_κp && succeed_xK && succeed_xg)
        warn_str = "At w = $(w), the model has not been solved for"
        to_warn  = Vector{String}(undef, 0)
        if succeed_κp
            push!(to_warn, "κp")
        end
        if succeed_xK
            push!(to_warn, "xK")
        end
        if succeed_xg
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
Q̂_calculation(stategrid::StateGrid, Q̂::AbstractVector{S}, μw::AbstractVector{S}, σw::AbstractVector{S},
    κw::AbstractVector{S}, rf::AbstractVector{S}, rg::AbstractVector{S}, rh::AbstractVector{S}, Q::S, λ::S;
    N_GH::Int = 10, tol::S = 1e-5, max_it::Int = 1000, Q̂_interp_method = Gridded(Linear()),
    testing::Bool = false) where {S <: Real}
```

calculates Q̂ using Gauss-Hermite quadrature.
"""
function Q̂_calculation(stategrid::StateGrid, Q̂::AbstractVector{S}, μw::AbstractVector{S}, σw::AbstractVector{S},
                       κw::AbstractVector{S}, rf::AbstractVector{S}, rg::AbstractVector{S}, rh::AbstractVector{S}, Q::S, λ::S;
                       N_GH::Int = 10, tol::S = 1e-5, max_it::Int = 1000, Q̂_interp_method = Gridded(Linear()),
                       testing::Bool = false) where {S <: Real}

    ## Set up
    spread = rf - rg
    ϵ_nodes, weight_nodes = gausshermite(N_GH)
    sort!(ϵ_nodes, rev = true) # to match the implementation by Maliar and Maliar (see the replication package for Li (2020))
    if testing
        err_vec = Vector{S}(undef, max_it)
    end

    ## Main calculation loop
    err     = 1.
    iter_no = 0
    while (err > tol) && (iter_no < max_it)
        iter_no += 1
        Q̂_last = copy(Q̂)
        Q̂_interp = interpolate(stategrid[:w], Q̂_last, Q̂_interp_method)

        # Calculate Q̂ everywhere except w = 0
        for i in 2:length(stategrid[:w])
            # Unpack objects characterizing equilibrium
            wᵢ  = stategrid[:w][i]
            μwᵢ = μw[i]
            σwᵢ = σw[i]
            κwᵢ = κw[i]

            # Calculate expectation
            w_noshock   = wᵢ .* (1. .+ μw * dt .+ σw .* ϵ_nodes)
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
    ∂p∂w0, ∂p∂wN = get_setting(m, :boundary_conditions)[:∂p∂w]

    f1 = function _ode_nojump_li2020(p, θ, w)
        if w == 0.
            ∂p∂w = ∂p∂w0
        elseif w == 1.
            ∂p∂w = ∂p∂wN
        else
            ψ = max(min((Φ(p, θ[:χ], θ[:δ]) + θ[:ρ] * p - θ[:AL]) / (θ[:AH] - θ[:AL]), 1.), 0.)
            xK = (ψ / w)
            yK = (1. - ψ) / (1. - w)
            σp = sqrt((θ[:AH] - θ[:AL]) / (p * (xK - yK))) - θ[:σK]
            σ  = xK * (θ[:σK] + σp)
            σh = yK * (θ[:σK] + σp)

            ∂p∂w = max(0., σp ./ (w .* (1. - w) .* (xK - yK) .* (θ[:σK] + σp)))
        end

        return ∂p∂w
    end

    # Terminal condition
    ode_condition(p, w, integrator) = p - get_setting(m, :boundary_conditions)[:p][2]
    ode_affect!(integrator) = terminate!(integrator)
    cb = ContinuousCallback(ode_condition, ode_affect!)

    return f1, cb
end
