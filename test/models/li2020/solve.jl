# This script tests the internal functions of eqcond.jl
using Test, OrderedCollections, HDF5
include("../../../src/includeall.jl")

verbose = :high # set to :none to avoid printing output
rp = joinpath(dirname(@__FILE__), "../../reference/models/li2020")

@testset "Solve Li2020" begin
    # Try with default method
    m = Li2020()
    vars_for_error = [:p, :Q̂]
    stategrid_default, funcvar_default, derivs_default, endo_default, flag_default = solve(m, verbose = verbose,
                                                                                           vars_for_error = vars_for_error)

    # Now try with different error approaches and ensure similar convergence
    m <= Setting(:error_method, :Linf)
    m <= Setting(:tol, 1e-5)
    stategrid_Linf, funcvar_Linf, derivs_Linf, endo_Linf, flag_Linf = solve(m, verbose = verbose,
                                                                            vars_for_error = vars_for_error)

    m <= Setting(:error_method, :L2)
    m <= Setting(:tol, 1e-10)
    stategrid_L2, funcvar_L2, derivs_L2, endo_L2, flag_L2 = solve(m, verbose = verbose,
                                                                  vars_for_error = vars_for_error)

    @test flag_default
    @test flag_Linf
    @test flag_L2
    for k in [:p, :Q̂]
        @test funcvar_default[k] ≈ funcvar_L2[k] atol=1e-5
        @test funcvar_default[k] ≈ funcvar_Linf[k] atol=1e-5
    end

    @test funcvar_default[:p] == h5read(joinpath(rp, "solve.h5"), "p")
    @test funcvar_default[:Q̂] == h5read(joinpath(rp, "solve.h5"), "Qhat")
end

#=
# Uncomment to re-write output
h5open(joinpath(rp, "solve.h5"), "w") do file
    write(file, "p", funcvar_default[:p])
    write(file, "Qhat", funcvar_default[:Q̂])
end
=#

nothing
