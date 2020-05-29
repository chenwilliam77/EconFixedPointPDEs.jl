using Test, ModelConstructors
include("../src/includeall.jl")


pvec = ParameterVector{Float64}(undef, 2)
pvec[1] = parameter(:a, 1.)
pvec[2] = parameter(:b, 2.)
@testset "Utility Functions" begin
    @test parameters_to_named_tuple(pvec) == (a = 1., b = 2.)
end
