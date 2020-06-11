function solve_slm_system(Mdes::AbstractMatrix{S}, rhs::AbstractVector{S},
                          Mreg::AbstractMatrix{S}, rhsreg::AbstractVector{S},
                          λ::S, Meq::AbstractMatrix{S}, rhseq::AbstractVector{S},
                          Mineq::AbstractMatrix{S}, rhsineq::AbstractVector{S};
                          method::Symbol = :ipopt, atol::S = NaN, rtol::S = NaN,
                          max_eval::Int = 0, max_time::S = 30.,
                          init::AbstractVector{S} = Vector{S}(undef, 0),
                          use_adnls::Bool = true, use_sparse::Bool = true,
                          verbose::Symbol = :low) where {S <: Real}

    # out_info = Dict{Symbol, Symbol}()

    Mfit = vcat(Mdes, Mreg .* λ)
    rhsfit = vcat(rhs, rhsreg .* λ)

    if verbose == :high
        println("Condition number of the regression: $(cond(Mdes))")
    end

    if isempty(Mineq) && isempty(Meq)
        # Backslash suffices in this case
        coef = Mfit \ rhsfit

        # out_info[:solver] = :backslash
    else
        # Set up NLPModel and constraints
        if isempty(Mineq)
            lcon = rhseq
            ucon = rhseq
            C = use_sparse ? sparse(Meq) : Meq
        else
            lcon = vcat(fill(-Inf, length(rhsineq)), rhseq) # -Inf for the inequalities since that's how lsqlin is written in MATLAB
            ucon = vcat(rhsineq, rhseq)
            C = use_sparse ? vcat(sparse(Mineq), sparse(Meq)) : vcat(Mineq, Meq)
        end

        if use_adnls
            if use_sparse
                Mdes = sparse(Mdes)
            end
            F(x) = Mdes * x - rhs

            if isempty(init)
                init = zeros(S, size(Mdes, 2))
            end

            nls = ADNLSModel(F, init, size(Mdes, 1), c = x -> C * x, lcon = lcon, ucon = ucon) # THIS WORKS BUT LETS TRY MANUALLY DEFINING THE HESS_STRUCTURE
        else
            error("Using LLSModel does not work yet.")
            nls = LLSModel(Mdes, rhs; C = C, lcon = lcon, ucon = ucon) # THIS WORKS BUT LETS TRY MANUALLY DEFINING THE HESS_STRUCTURE
        end

        # Decide solver for minimizing the constrained linear least-squares problem
        if method == :ipopt
            coef = ipopt(nls; print_level = verbose == :high ? 3 : 0).solution
        else
            error("Method $method not is not implemented.")
        end

        # out_info[:solver] = :NLPModels
    end

    if verbose == :high
        println("Solver employed: $(out_info[:solver])")
        if !isempty(Mineq)
            println(nls)
        end
    end

    return coef
end
