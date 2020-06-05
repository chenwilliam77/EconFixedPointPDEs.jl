function solve_slm_system(Mdes::AbstractMatrix{S} rhs::AbstractVector{S},
                          Mreg::AbstractMatrix{S}, rhsreg::AbstractVector{S},
                          λ::S, Meq::AbstractMatrix{S}, rhseq::AbstractVector{S},
                          Mineq::AbstractMatrix{S}, rhsineq::AbstractVector{S};
                          method::Symbol = :ipopt, atol::S = NaN, rtol::S = NaN,
                          max_eval::Int = 0, max_time::S = 30.,
                          verbose::Symbol = :low) where {S <: Real}

    info_out = Dict{Symbol, Symbol}()

    Mfit = vcat(Mdes, Mreg .* λ)
    rhsfit = vcat(rhs, rhsreg .* λ)

    if verbose == :high
        println("Condition number of the regression: $(cond(Mdes))")
    end

    if isempty(Mineq) && isempty(Meq)
        # Backslash suffices in this case
        coef = Mfit \ rhsfit
        out_info[:solver] = :backslash
    elseif isempty(Mineq)
        # Use LinearLeastSquares.jl
        coef        = Variable(length(rhs))
        constraints = Meq * coef == rhseq
        objective   = sum_squares(Mdes * coef - rhs)
        minimize!(objective, constraints)

        out_info[:solver] = :LinearLeastSquares
    else
        # Use NLPModel
        lcon = vcat(fill(-Inf, length(rhsineq)), rhseq) # -Inf for the inequalities since that's how lsqlin is
        rcon = vcat(rhsineq, rhseq) #
        n_ineq = size(Mineq, 1)
        n_eq   = size(Meq,   1)
        C = spzeros(S, n_ineq + n_eq, n_ineq + n_eq)
        C[1:n_ineq, 1:n_ineq] = Mineq
        C[1:n_eq,   1:n_eq]   = Meq

        nls = LLSModel(Mdes, rhs; C = C, lcon = lcon, rcon = rcon)
        if method == :ipopt
            coef = ipopt(nlp).solution
        elseif method == :tron || method == :trust_region
            coef = tron(nlp).solution
        else
            error("Method $method not is not implemented.")
        end

        out_info[:solver] = :NLPModels
    end

    if verbose == :high
        println("Solver employed: $(out_info[:solver])")
        if !isempty(Mineq)
            println(nls)
        end
    end

    return coef
end
