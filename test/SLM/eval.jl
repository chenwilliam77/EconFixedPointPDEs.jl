using Test, ModelConstructors
include(joinpath(dirname(@__FILE__), "../../src/includeall.jl"))

# TEST USING SLM OBJECT THAT IT MATCHES THE DESIRED OUTPUT
rp = joinpath(dirname(@__FILE__), "../reference/SLM")
in_inc = load(joinpath(rp, "eval.jld2"))
in_dec = load(joinpath(rp, "eval_decrease.jld2"))
in_sin = load(joinpath(rp, "eval_sine.jld2"))
in_sinnoineq = load(joinpath(rp, "eval_sinnoineq.jld2"))

slms = Vector{AbstractSLM}(undef, 0)

for (i, in_data) in zip(1:4, [in_inc, in_dec, in_sin, in_sinnoineq])
    if i == 1
        push!(slms, SLM(vec(in_data["w"]), vec(in_data["p_sol"]); knots = Int(in_data["knots"]),
                        increasing = true, concave_down = true, left_value = in_data["p0_norun"],
                        right_value = in_data["p1_norun"], min_value = in_data["p0_norun"] - 1e-3,
                        max_value = in_data["p1_norun"] + 1e-3))
    elseif i == 2
        push!(slms, SLM(vec(in_data["w"]), vec(in_data["rev_p_sol"]); knots = Int(in_data["knots"]),
                        decreasing = true, concave_up = true, right_value = in_data["p0_norun"],
                        left_value = in_data["p1_norun"], min_value = in_data["p0_norun"] - 1e-3,
                        max_value = in_data["p1_norun"] + 1e-3))
    elseif i == 3
        push!(slms, SLM(vec(in_data["x"]), vec(in_data["y"]); knots = Int(in_data["knots"]),
                        increasing_intervals = in_data["inc_int"], decreasing_intervals = in_data["dec_int"],
                        concave_up_intervals = in_data["cu_int"], concave_down_intervals = in_data["cd_int"],
                        right_value = in_data["rightvalue"], left_value = in_data["leftvalue"]))
    else
        push!(slms, SLM(vec(in_data["x"]), vec(in_data["y"]); knots = Int(in_data["knots"]),
                        right_value = in_data["rightvalue"], left_value = in_data["leftvalue"]))
    end
end

@testset "eval of SLM" begin
    for (i, in_data) in zip(1:4, [in_inc, in_dec, in_sin, in_sinnoineq])
        if i == 1
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["w"])) in_data["phat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["w"]), 0) in_data["phat"]
        elseif i == 2
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["w"])) in_data["phat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["w"]), 0) in_data["phat"]
        elseif i == 3
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"])) in_data["yhat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"]), 0) in_data["yhat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"]), 1) in_data["yphat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"]), 2) in_data["ypphat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"]), 3) in_data["yppphat"]
        else
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"])) in_data["yhat"]
            @test @test_matrix_approx_eq eval(slms[i], vec(in_data["xhat"]), 0) in_data["yhat"]
        end
    end
end
