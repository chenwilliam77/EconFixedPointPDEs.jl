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
@testset "Scaling" begin
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
end

# Design matrix
Mdes = Dict()
rhs = Dict()
Mineq = Dict()
rhsineq = Dict()
Meq = Dict()
rhseq = Dict()
for (name, in_data) in zip([:inc, :dec, :sin], [in_inc, in_dec, in_sin])
    Mineq[name] = zeros(0, Int(in_data["nc"]))
    Meq[name] = zeros(0, Int(in_data["nc"]))
    rhsineq[name] = Vector{Float64}(undef, 0)
    rhseq[name] = Vector{Float64}(undef, 0)
    Mdes[name] = construct_design_matrix(vec(in_data["x"]), vec(in_data["knots"]),
                                         vec(in_data["dx"]), Int.(vec(in_data["xbin"])),
                                         Int(in_data["nx"]), Int(in_data["nk"]), Int(in_data["nc"]))
    rhs[name] = in_data["y"]
end
@testset "Design matrix" begin
    for (name, in_data) in zip([:inc, :dec, :sin], [in_inc, in_dec, in_sin])
        @test @test_matrix_approx_eq Mdes[name] in_data["Mdes"]
        @test rhs[name] == in_data["rhs"]
    end
end


# Regularizer matrix
Mreg = Dict()
for (name, in_data) in zip([:inc, :dec, :sin], [in_inc, in_dec, in_sin])
    Mreg[name] = construct_regularizer(vec(in_data["dx"]), Int(in_data["nk"]))
end
reg_inc = load(joinpath(rp, "regularizer.jld2"))
reg_dec = load(joinpath(rp, "regularizer_decrease.jld2"))
reg_sin = load(joinpath(rp, "regularizer_sine.jld2"))
@testset "Regularizer" begin
    for (name, reg_data) in zip([:inc, :dec, :sin], [reg_inc, reg_dec, reg_sin])
        @test @test_matrix_approx_eq Mreg[name] reg_data["Mreg"]
        @test all(reg_data["rhsreg"] .== 0.)
    end
end

# Test C2
C2_inc = load(joinpath(rp, "C2.jld2"))
C2_dec = load(joinpath(rp, "C2_decrease.jld2"))
C2_sin = load(joinpath(rp, "C2_sine.jld2"))
for (name, in_data) in zip([:inc, :dec, :sin], [in_inc, in_dec, in_sin])
    Meq[name], rhseq[name] = C2_matrix(Int(in_data["nk"]), Int(in_data["nc"]), vec(in_data["dx"]), Meq[name], rhseq[name])
end
@testset "C2" begin
    for (name, C2_data) in zip([:inc, :dec, :sin], [C2_inc, C2_dec, C2_sin])
        @test @test_matrix_approx_eq Meq[name] C2_data["Meq"]
        @test @test_matrix_approx_eq rhseq[name] C2_data["rhseq"]
    end
end

# Left and right values
left_inc = load(joinpath(rp, "left_value.jld2"))
left_dec = load(joinpath(rp, "left_value.jld2")) # did not create a "decreasing" version, so we just use the same file as increasing
left_sin = load(joinpath(rp, "left_value_sine.jld2")) # and adjust accordingly, noting that the left and right values are the reverse ones
right_inc = load(joinpath(rp, "right_value.jld2"))
right_dec = load(joinpath(rp, "right_value.jld2"))
right_sin = load(joinpath(rp, "right_value_sine.jld2"))
for (i, name, side_data, in_data) in zip(1:3, [:inc, :dec, :sin], [left_inc, right_dec, left_sin], [in_inc, in_dec, in_sin])
    if i == 2
        Meq[name], rhseq[name] = set_right_value(side_data["right_value"], Int(in_data["nc"]), Int(in_data["nk"]), Meq[name], rhseq[name])
    else
        Meq[name], rhseq[name] = set_left_value(side_data["left_value"], Int(in_data["nc"]), Meq[name], rhseq[name])
    end
end
for (i, name, side_data, in_data) in zip(1:3, [:inc, :dec, :sin], [right_inc, left_dec, right_sin], [in_inc, in_dec, in_sin])
    if i == 2
        Meq[name], rhseq[name] = set_right_value(side_data["left_value"], Int(in_data["nc"]), Meq[name], rhseq[name])
    else
        Meq[name], rhseq[name] = set_right_value(side_data["right_value"], Int(in_data["nc"]), Int(in_data["nk"]), Meq[name], rhseq[name])
    end
end
@testset "Left and right value" begin
    for (name, in_data) in zip([:inc, :sin], [right_inc, right_sin])
        @test @test_matrix_approx_eq Meq[name] in_data["Meq"]
        @test @test_matrix_approx_eq rhseq[name] in_data["rhseq"]
    end
end

# Left and right values
left_inc = load(joinpath(rp, "left_value.jld2"))
left_dec = load(joinpath(rp, "left_value.jld2")) # did not create a "decreasing" version, so we just use the same file as increasing
left_sin = load(joinpath(rp, "left_value_sine.jld2")) # and adjust accordingly, noting that the left and right values are the reverse ones
