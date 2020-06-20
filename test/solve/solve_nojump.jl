using Test, OrdinaryDiffEq, HDF5, ModelConstructors
include(joinpath(dirname(@__FILE__), "../../src/includeall.jl"))

m = Li2020()
stategrid, functional_variables, derivatives, endogenous_variables = solve(m; nojump = true, nojump_method = :ode)

@testset "Solving the no jump equilibrium using Li2020" begin
    truep    = h5read(joinpath(dirname(@__FILE__), "../reference/models/li2020/solve_nojump_li2020_eqm.h5"), "p")
    truedpdw = h5read(joinpath(dirname(@__FILE__), "../reference/models/li2020/solve_nojump_li2020_eqm.h5"), "dpdw")
    @test @test_matrix_approx_eq truep functional_variables[:p]
    @test @test_matrix_approx_eq truedpdw derivatives[:∂p∂w]

    for (k, v) in endogenous_variables
        trueval = h5read(joinpath(dirname(@__FILE__), "../reference/models/li2020/solve_nojump_li2020_eqm.h5"), string(detexify(k)))
        @test @test_matrix_approx_eq trueval v
    end
end
