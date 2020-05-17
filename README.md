# EconFixedPointPDEs
This repository aims to extend [EconPDEs.jl](https://github.com/matthieugomez/EconPDEs.jl) to permit jump diffusions
in endogenous state variables by applying pseudo-transient continuation to the value function iteration method
developed by [Li, Wenhao (2020)
"Public Liquidity and Financial Crises"](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3175101).
The key problem is that with jump diffusions the Hamilton-Jacobi-Bellman equation becomes an integro-differential equation.
Li (2020) handles this problem with tempered fixed-point iteration.
The method, however, requires log utility for agents. To extend the method for more general preferences and models,
I plan to use pseudo-transient continuation as the proposal method for the next guess of value functions in the
tempered fixed-point iteration.


The current plan of development is

1. Replicate Li (2020) in Julia.
2. Rewrite the solution method to apply pseudo-transient continuation.
3. Extend EconPDEs.jl to permit generic problems featuring fixed points.