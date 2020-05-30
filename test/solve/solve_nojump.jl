using Test, DifferentialEquations, HDF5, ModelConstructors
include("../../src/includeall.jl")

m = Li2020()
stategrid, differential_variables, endogenous_variables = solve(m; nojump = true)

@testset "Solving the no jump equilibrium using Li2020" begin
    truep    = h5read("../reference/solve_nojump_li2020_eqm.h5", "p")
    truedpdw = h5read("../reference/solve_nojump_li2020_eqm.h5", "dpdw")
    @test @test_matrix_approx_eq truep differential_variables[:p]
    @test @test_matrix_approx_eq truedpdw differential_variables[:∂p∂w]

    for (k, v) in endogenous_variables
        trueval = h5read("../reference/solve_nojump_li2020_eqm.h5", string(detexify(k)))
        @test @test_matrix_approx_eq trueval v
    end
end
