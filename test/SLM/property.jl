using Test, FileIO, ModelConstructors
include("../../src/includeall.jl")
rp = "../reference/SLM" # reference path

# Set up inputs
input_inc = load(joinpath(rp, "find_solution_input.jld2"))
input_dec = load(joinpath(rp, "find_solution_input_decrease.jld2"))
input_sin = load(joinpath(rp, "find_solution_input_sine.jld2"))
in_inc = load(joinpath(rp, "design.jld2"))
in_dec = load(joinpath(rp, "design_decrease.jld2"))
in_sin = load(joinpath(rp, "design_sine.jld2"))
d = Dict()
default_slm_kwargs!(d)
d[:left_value] = input_inc["p0_norun"]
d[:right_value] = input_inc["p1_norun"]

# Scale problem
d_false = deepcopy(d)
d_false[:scaling] = false
yscale_inc = load(joinpath(rp, "scaleproblem.jld2"))
yscale_dec = load(joinpath(rp, "scaleproblem_decrease.jld2"))
yscale_sin = load(joinpath(rp, "scaleproblem_sine.jld2"))
leftval_inc = load(joinpath(rp, "left_value.jld2"))
rightval_inc = load(joinpath(rp, "right_value.jld2"))
for (i, in_data, yscale_data) in zip(1:3, [input_inc, input_dec, input_sin], [yscale_inc, yscale_dec, yscale_sin])
    if i == 1
        ŷ = scale_problem!(vec(in_data["w"]), vec(in_data["p_sol"]), d)
    elseif i == 2
        ŷ = scale_problem!(vec(in_data["w"]), vec(in_data["rev_p_sol"]), d)
    else
        ŷ = scale_problem!(vec(in_data["x"]), vec(in_data["y"]), d)
    end
    if i == 1
        @test d[:left_value] == leftval_inc["left_value"]
        @test d[:right_value] == rightval_inc["right_value"]
    end

    @test d[:y_scale] == yscale_data["YScale"]
    @test d[:y_shift] == yscale_data["YShift"]
    @test ŷ           == vec(yscale_data["yhat"])
end
@test vec(in_inc["y"]) == scale_problem!(vec(in_inc["x"]), vec(in_inc["y"]), d_false)
@test d_false[:left_value] == input_inc["p0_norun"]
@test d_false[:right_value] == input_inc["p1_norun"]
@test d_false[:y_shift] == 0.
@test d_false[:y_scale] == 1.

# Test C2
C2_inc = load(joinpath(rp, "C2.jld2"))
C2_dec = load(joinpath(rp, "C2_decrease.jld2"))
C2_sin = load(joinpath(rp, "C2_sine.jld2"))
@test @test_matrix_approx_eq C2_matrix(Int(in_inc["nk"]), Int(in_inc["nc"]), vec(in_inc["dx"])) C2_inc["MC2"]
