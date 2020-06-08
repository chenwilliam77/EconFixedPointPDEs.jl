using Test, FileIO
include("../../src/includeall.jl")


mat_inc = load("../reference/SLM/histc.jld2")
mat_dec = load("../reference/SLM/histc.jld2")
mat_sin = load("../reference/SLM/histc.jld2")

@testset "bin_sort" begin
    for mat_dic in [mat_inc, mat_dec, mat_sin]
        outbins = bin_sort(vec(mat_dic["x"]), vec(mat_dic["knots"]))
        @test outbins == Int.(vec(mat_dic["xbin"]))
    end
end

@testset "Other utility functions for SLM" begin
    x = (0.:.1:1.) .^ 2
    @test choose_knots(10, x) == collect(range(0., stop = 1., length = 10))
end
