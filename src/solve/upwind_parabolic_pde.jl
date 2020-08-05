# Translation of code from Yuliy Sannikov, payoff_policy_growth, with the modification that instead of S, we plug in S²
function upwind_parabolic_pde(X, R, μ, Σ², G, V, dt_div_1pdt)
    N = length(X)
    dX = diff(X)

    # Perform upwind scheme w/centered difference on diffusion term
    Σ²0 = zeros(N)
    Σ²0[2:N-1] .= Σ²[2:N-1] ./ (dX[1:N-2] + dX[2:N-1]) # approx Σ² / (2 * dx): this term is the Σ²/2 coefficient
    DU = -(max.(μ[1:N-1], 0.) + Σ²0[1:N-1]) ./ dX .* dt_div_1pdt # up diagonal, μ divided by dX, Σ²0 is Σ² / (2 * dx^2), th
    DD = -(max.(-μ[2:N], 0.) + Σ²0[2:N]) ./ dX .* dt_div_1pdt # down diagonal, note should be negative b/c FD scheme makes DD negative, multiplied by negative drift ⇒ positive, then subtracted ⇒ negative

    # observe: Σ² and μ are zero at endpoints, hence Σ²0 zero at endpts too ->
    # boundary conditions for our PDE

    D0 = (1 - dt_div_1pdt) .* ones(N) + dt_div_1pdt .* R # diagonal
    D0[1:N-1] = D0[1:N-1] - DU
    D0[2:N] = D0[2:N] - DD # subtract twice b/c centered diff
    A = spdiagm(0 => D0, 1 => DU, -1 => DD) # + spdiagm(DU,1,N,N) + spdiagm(DD[1:N-1],-1,N,N)
    F = A \ (G .* dt_div_1pdt + V .* (1 - dt_div_1pdt)) # solve linear system

#=    # Equivalent to this code, which constructs the first and second finite difference matrices separately
    DU1 = zeros(N)
    DU2 = zeros(N)
    DD1 = zeros(N)
    DD2 = zeros(N)
    D01 = zeros(N)
    D02 = zeros(N)    DU1[2:N] = max.(μ[1:N - 1], 0.) ./ dX
    DU2[2:N] = Σ²0[1:N - 1] ./ dX
    DD1[1:N - 1] = max.(-μ[2:N], 0.) ./ dX
    DD2[1:N - 1] = Σ²0[2:N] ./ dX
    D01[1:N - 1] .= -DU1[2:N]
    D01[2:N] .-= DD1[1:N - 1]
    D02[1:N - 1] .= -DU2[2:N]
    D02[2:N] .-= DD2[1:N - 1]
    L1 = spdiagm(0 => D01, -1 => DD1[1:N - 1], 1 => DU1[2:N])
    L2 = spdiagm(0 => D02, -1 => DD2[1:N - 1], 1 => DU2[2:N])
    Acheck = spdiagm(0 => (1 - dt_div_1pdt) .* ones(N) + dt_div_1pdt .* R) - dt_div_1pdt .* (L1 + L2)=#

    return F
end
