using Test
include(joinpath(dirname(@__FILE__), "../../src/includeall.jl"))

d = Dict()
default_slm_kwargs!(d)
d1 = deepcopy(d)
d1[:increasing] = true

@testset "Default SLM keyword arguments" begin
    for (i, di) in enumerate([d, d1])
        @test di[:degree] == 3
        @test di[:scaling] == true
        @test di[:knots] == 6
        @test di[:C2] == true
        @test di[:λ] == 1e-4
        if i == 1
            @test di[:increasing] == false
        else
            @test di[:increasing] == true
        end
        @test di[:decreasing] == false
        @test isempty(di[:increasing_intervals])
        @test isempty(di[:decreasing_intervals])
        @test isa(di[:increasing_intervals], Matrix{Float64})
        @test isa(di[:decreasing_intervals], Matrix{Float64})
        @test di[:concave_up] == false
        @test di[:concave_down] == false
        @test isempty(di[:concave_up_intervals])
        @test isempty(di[:concave_down_intervals])
        @test isa(di[:concave_up_intervals], Matrix{Float64})
        @test isa(di[:concave_down_intervals], Matrix{Float64})
        @test isnan(di[:left_value])
        @test isnan(di[:right_value])
        @test isnan(di[:min_value])
        @test isnan(di[:max_value])
        @test di[:min_max_sample_points] == [.017037, .066987, .1465, .25, .37059,
                                             .5, .62941, .75, .85355, .93301, .98296]
        @test isempty(di[:init])
        @test di[:use_sparse] == false
        @test di[:use_lls]
    end
end
