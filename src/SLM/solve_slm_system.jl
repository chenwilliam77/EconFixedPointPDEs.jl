"""
```
function solve_slm_system(Mdes::AbstractMatrix{S}, rhs::AbstractVector{S},
                          Mreg::AbstractMatrix{S}, rhsreg::AbstractVector{S},
                          λ::S, Meq::AbstractMatrix{S}, rhseq::AbstractVector{S},
                          Mineq::AbstractMatrix{S}, rhsineq::AbstractVector{S};
                          method::Symbol = :ipopt, atol::S = NaN, rtol::S = NaN,
                          max_iter::Int = 0, max_time::S = 30.,
                          init::AbstractVector{S} = Vector{S}(undef, 0),
                          use_lls::Bool = true, use_sparse::Bool = true,
                          verbose::Symbol = :low) where {S <: Real}
```

estimates the coefficient of a least squares spline by solving the
constrained linear least squares problem

```
min ½ || Mfit * x - rhsfit ||²
```

subject to the constraints

```
Meq * x = rhseq,    Mineq * x ≤ rhsineq,
```

where `Mfit = vcat(Mdes, λ .* Mreg)` and `rhsfit = vcat(rhs, λ .* rhsreg)`.

### Inputs
- `Mdes`: design matrix in system of equations defining the spline
- `rhs`: RHS of the system of equations defining the spline
- `Mreg`: matrix in system of equations regularizing the spline's smoothness
- `rhsreg`: RHS of the system of equations regularizing the spline smoothnes
- `λ`: degree of smoothness (smaller ⇒ smoother)
- `Meq`: matrix for equality constraints
- `rhseq`: the RHS of equality constraints
- `Mineq`: matrix for inequality constraints
- `rhsineq`: upper bound for inequality constraints

### Keywords
- `method`: minimization algorithm, currently only the interior point optimizer is used (Ipopt).
    New methods are welcome! One algorithm that may be added is adapting the matrices to the trust-region method
    via the calculation of pseudo-inverses
- `atol`: absolute tolerance for minimization algorithm
- `rtol`: relative tolerance for minimization algorithm
- `max_iter`: maximum number of evaluations
- `max_time`: maximum time (in minutes) for optimization
- `init`: initial guess of coefficients for the spline
- `use_lls`: use an `ADNLSModel` to define the problem. Currently, using an `LLSModel` (or equivalent) does not work.
    See the `NLPModels.jl` package.
- `use_sparse`: use sparse matrices (assuming the inputs are not already sparse matrices)
- `verbose`: how much information will be printed. Can be `:low` or `:high`
"""
function solve_slm_system(Mdes::AbstractMatrix{S}, rhs::AbstractVector{S},
                          Mreg::AbstractMatrix{S}, rhsreg::AbstractVector{S},
                          λ::S, Meq::AbstractMatrix{S}, rhseq::AbstractVector{S},
                          Mineq::AbstractMatrix{S}, rhsineq::AbstractVector{S};
                          method::Symbol = :ipopt, atol::S = 1e-8, rtol::S = 1e-6,
                          max_iter::Int = 3000, max_time::S = 3600.,
                          init::AbstractVector{S} = Vector{S}(undef, 0),
                          use_lls::Bool = true, use_sparse::Bool = true,
                          verbose::Symbol = :low, kwargs...) where {S <: Real}

    if use_sparse && all([issparse(x) for x in [Mdes, Mreg, Meq, Mineq]])
        # Don't spend time converting already sparse matrices
        use_sparse = false
    end

    Mfit = use_sparse ? vcat(sparse(Mdes), sparse(Mreg) .* λ) : vcat(Mdes, Mreg .* λ)
    rhsfit = vcat(rhs, rhsreg .* λ)

    if verbose == :high
        println("Condition number of the regression: $(cond(Mdes))")
    end

    if isempty(Mineq) && isempty(Meq)
        # Backslash suffices in this case
        coef = Mfit \ rhsfit
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

        if use_lls
            nls = LLSModel(Mfit, rhsfit; C = C, lcon = lcon, ucon = ucon)
        else
            if isempty(init)
                init = zeros(S, size(Mfit, 2))
            end

            nls = ADNLSModel(x -> Mfit * x - rhsfit, init, size(Mfit, 1), c = x -> C * x, lcon = lcon, ucon = ucon)
        end

        # Decide solver for minimizing the constrained linear least-squares problem
        if method == :ipopt
            if use_lls
                # Feasibility Form adds additional coefficient which we don't want, hence
                # the [1:size(Mdes, 2)] slicing.
                coef = ipopt(FeasibilityFormNLS(nls); print_level = verbose == :high ? 3 : 0,
                             tol = atol, acceptable_tol = rtol, max_iter = max_iter,
                             max_cpu_time = max_time, kwargs...).solution[1:size(Mdes, 2)]
            else
                #
                coef = ipopt(nls; print_level = verbose == :high ? 3 : 0, tol = atol, acceptable_tol = rtol,
                             max_iter = max_iter, max_cpu_time = max_time, kwargs...).solution
            end
        else
            error("Method $method not is not implemented.")
        end
    end

    if verbose == :high
        println("Solver employed: $(out_info[:solver])")
        if !isempty(Mineq)
            println(nls)
        end
    end

    return coef
end
