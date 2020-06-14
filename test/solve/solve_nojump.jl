using Test, OrdinaryDiffEq, HDF5, ModelConstructors
include("../../src/includeall.jl")

m = Li2020()
stategrid, functional_variables, derivatives, endogenous_variables = solve(m; nojump = true, use_ode = true)

@testset "Solving the no jump equilibrium using Li2020" begin
    truep    = h5read("../reference/solve_nojump_li2020_eqm.h5", "p")
    truedpdw = h5read("../reference/solve_nojump_li2020_eqm.h5", "dpdw")
    @test @test_matrix_approx_eq truep functional_variables[:p]
    @test @test_matrix_approx_eq truedpdw derivatives[:∂p∂w]

    for (k, v) in endogenous_variables
        trueval = h5read("../reference/solve_nojump_li2020_eqm.h5", string(detexify(k)))
        @test @test_matrix_approx_eq trueval v
    end
end
