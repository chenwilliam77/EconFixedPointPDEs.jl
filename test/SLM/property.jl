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
        Meq[name], rhseq[name] = set_left_value(side_data["left_value"], Int(in_data["nc"]), Meq[name], rhseq[name])
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

# Global minimum and maximum
minmax_inc = load(joinpath(rp, "min_max.jld2"))
minmax_dec = load(joinpath(rp, "min_max_decrease.jld2"))
minmax_sin = load(joinpath(rp, "min_max_sine.jld2"))
for (name, in_data, minmax_data) in zip([:inc, :dec], [in_inc, in_dec], [minmax_inc, minmax_dec])
    Mineq[name], rhsineq[name] = set_min_value(minmax_data["min_value"], Int(in_data["nk"]), Int(in_data["nc"]),
                                               vec(in_data["dx"]), Mineq[name], rhsineq[name])
    Mineq[name], rhsineq[name] = set_max_value(minmax_data["max_value"], Int(in_data["nk"]), Int(in_data["nc"]),
                                               vec(in_data["dx"]), Mineq[name], rhsineq[name])
end
@testset "Global minimum and maximum" begin
    for (name, minmax_data) in zip([:inc, :dec], [minmax_inc, minmax_dec])
        @test @test_matrix_approx_eq Mineq[name]  minmax_data["Mineq"]
        @test @test_matrix_approx_eq rhsineq[name] vec(minmax_data["rhsineq"])
    end
end

# Monotonicity
mono_inc = load(joinpath(rp, "monotone.jld2"))
mono_dec = load(joinpath(rp, "monotone_decrease.jld2"))
mono_sin = load(joinpath(rp, "monotone_sine.jld2"))
inc_int = [0 π/2; 3*π/2 5*π/2; 7*π/2 4*pi]
dec_int = [π/2 3*π/2; 5*π/2 7*π/2]
for (i, name, in_data, mono_data) in zip(1:3, [:inc, :dec, :sin], [in_inc, in_dec, in_sin], [mono_inc, mono_dec, mono_sin])
    monotone_settings = Vector{NamedTuple{(:knotlist, :increasing), Tuple{Tuple{Int, Int}, Bool}}}(undef, 0)
    total_monotone_intervals = 0
    if i == 1
        total_monotone_intervals += monotone_increasing!(monotone_settings, Int(in_data["nk"]))
    elseif i == 2
        total_monotone_intervals += monotone_decreasing!(monotone_settings, Int(in_data["nk"]))
    elseif i == 3
        total_monotone_intervals += increasing_intervals_info!(monotone_settings, vec(in_data["knots"]), inc_int, Int(in_data["nk"]))
        total_monotone_intervals += decreasing_intervals_info!(monotone_settings, vec(in_data["knots"]), dec_int, Int(in_data["nk"]))
    end
    Mineq[name], rhsineq[name] = construct_monotonicity_matrix(monotone_settings, Int(in_data["nc"]), Int(in_data["nk"]),
                                                              vec(in_data["dx"]), total_monotone_intervals, Mineq[name], rhsineq[name])
end
@testset "Monotonicity" begin
    for (name, mono_data) in zip([:inc, :dec, :sin], [mono_inc, mono_dec, mono_sin])
        @test @test_matrix_approx_eq Mineq[name]  mono_data["Mineq"]
        @test @test_matrix_approx_eq rhsineq[name] mono_data["rhsineq"]
    end
end

# Curvature
curv_inc = load(joinpath(rp, "curvature.jld2"))
curv_dec = load(joinpath(rp, "curvature_decrease.jld2"))
curv_sin = load(joinpath(rp, "curvature_sine.jld2"))
cu_int  = [π 2*π; 3*π 4*π]
cd_int  = [0 π; 2*π 3*π]
for (i, name, in_data) in zip(1:3, [:inc, :dec, :sin], [in_inc, in_dec, in_sin])
    curvature_settings = Vector{NamedTuple{(:concave_up, :range), Tuple{Bool, Vector{Float64}}}}(undef, 0)

    if i == 1
        concave_down_info!(curvature_settings)
    elseif i == 2
        concave_up_info!(curvature_settings)
    elseif i == 3
        concave_up_intervals_info!(curvature_settings, cu_int)
        concave_down_intervals_info!(curvature_settings, cd_int)
    end
    println( curvature_settings)
    Mineq[name], rhsineq[name] = construct_curvature_matrix(curvature_settings, Int(in_data["nc"]), Int(in_data["nk"]), vec(in_data["knots"]),
                                                              vec(in_data["dx"]), Mineq[name], rhsineq[name])
end
@testset "Curvature" begin
    for (name, curv_data) in zip([:inc, :dec, :sin], [curv_inc, curv_dec, curv_sin])
        @test @test_matrix_approx_eq Mineq[name]  curv_data["Mineq"]
        @test @test_matrix_approx_eq rhsineq[name] vec(curv_data["rhsineq"])
    end
end
