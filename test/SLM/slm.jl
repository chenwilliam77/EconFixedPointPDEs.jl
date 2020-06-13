using Test, FileIO
include("../../src/includeall.jl")

rp = "../reference/SLM"
in_inc = load(joinpath(rp, "find_solution_input.jld2"))
in_dec = load(joinpath(rp, "find_solution_input_decrease.jld2"))
in_sin = load(joinpath(rp, "find_solution_input_sine.jld2"))
in_sinnoineq = load(joinpath(rp, "find_solution_input_sinenoineq.jld2"))
out_inc = load(joinpath(rp, "slm_rescale.jld2"))
out_dec = load(joinpath(rp, "slm_rescale_decrease.jld2"))
out_sin = load(joinpath(rp, "slm_rescale_sine.jld2"))
out_sinnoineq = load(joinpath(rp, "slm_rescale_sinenoineq.jld2"))
cu_int  = [π 2*π; 3*π 4*π]
cd_int  = [0 π; 2*π 3*π]
inc_int = [0 π/2; 3*π/2 5*π/2; 7*π/2 4*pi]
dec_int = [π/2 3*π/2; 5*π/2 7*π/2]
test_inc = load(joinpath(rp, "solve_slm_system.jld2"))
scale_inc = load(joinpath(rp, "scaleproblem.jld2"))

@testset "SLM estimation" begin
    for (i, in_data, out_data) in zip(1:4, [in_inc, in_dec, in_sin, in_sinnoineq], [out_inc, out_dec, out_sin, out_sinnoineq])
        if i == 1
            slm = SLM(vec(in_data["w"]), vec(in_data["p_sol"]); knots = Int(in_data["knots"]),
                      increasing = true, concave_down = true, left_value = in_data["p0_norun"],
                      right_value = in_data["p1_norun"], min_value = in_data["p0_norun"] - 1e-3,
                      max_value = in_data["p1_norun"] + 1e-3)
        elseif i == 2
            slm = SLM(vec(in_data["w"]), vec(in_data["rev_p_sol"]); knots = Int(in_data["knots"]),
                      decreasing = true, concave_up = true, right_value = in_data["p0_norun"],
                      left_value = in_data["p1_norun"], min_value = in_data["p0_norun"] - 1e-3,
                      max_value = in_data["p1_norun"] + 1e-3)
        elseif i == 3
            slm = SLM(vec(in_data["x"]), vec(in_data["y"]); knots = Int(in_data["knots"]),
                      increasing_intervals = inc_int, decreasing_intervals = dec_int,
                      concave_up_intervals = cu_int, concave_down_intervals = cd_int,
                      right_value = 0., left_value = 0.)
        else
            slm = SLM(vec(in_data["x"]), vec(in_data["y"]); knots = Int(in_data["knots"]),
                      right_value = 0., left_value = 0.)
        end
        if i == 1 || i == 2
            @test maximum(abs.(get_coef(slm) - out_data["rescale_coef"])) < 5e-4
        else
            @test @test_matrix_approx_eq get_coef(slm) out_data["rescale_coef"]
        end
    end
end

#=
println("No sparse matrices")
@btime begin
    SLM(vec(in_sin["x"]), vec(in_sin["y"]); knots = Int(in_sin["knots"]),
        increasing_intervals = inc_int, decreasing_intervals = dec_int,
        concave_up_intervals = cu_int, concave_down_intervals = cd_int,
        right_value = 0., left_value = 0.)
end
println("Sparse matrices")
@btime begin
    SLM(vec(in_sin["x"]), vec(in_sin["y"]); knots = Int(in_sin["knots"]),
        increasing_intervals = inc_int, decreasing_intervals = dec_int,
        concave_up_intervals = cu_int, concave_down_intervals = cd_int,
        right_value = 0., left_value = 0., use_sparse = true)
end
=#
