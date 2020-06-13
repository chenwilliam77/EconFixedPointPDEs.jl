using Test, ModelConstructors
include("../src/includeall.jl")


pvec = ParameterVector{Float64}(undef, 2)
pvec[1] = parameter(:a, 1.)
pvec[2] = parameter(:b, 2.)
@testset "Utility Functions" begin
    @test parameters_to_named_tuple(pvec) == (a = 1., b = 2.)
    subs = [CartesianIndex(i, j) for (i, j) in zip(1:4, 3:6)]
    vals = [13.0, 24.4, 31.9, 14.0998898]
    @test maximum(abs.(sparse_accumarray(subs, vals, (5, 10)) - accumarray(subs, vals, (5, 10)))) < 1e-15
end
