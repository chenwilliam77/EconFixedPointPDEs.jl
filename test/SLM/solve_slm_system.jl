using Test, FileIO
include("../../src/includeall.jl")

# Load in input data
rp = "../reference/SLM/"
in_inc = load(joinpath(rp, "solve_slm_system.jld2"))
in_dec = load(joinpath(rp, "solve_slm_system_decrease.jld2"))
in_sin = load(joinpath(rp, "solve_slm_system_sine.jld2"))
in_sinnoineq = load(joinpath(rp, "solve_slm_system_sinenoineq.jld2"))
noconstraints = load(joinpath(rp, "solve_slm_system_noconstraints.jld2"))

#=println("No sparse matrices")
@btime begin
    solve_slm_system(in_inc["Mdes"], vec(in_inc["rhs"]), in_inc["Mreg"],
                     vec(in_inc["rhsreg"]), in_inc["finalRP"], in_inc["Meq"], vec(in_inc["rhseq"]),
                     in_inc["Mineq"], vec(in_inc["rhsineq"]))
end

println("Sparse matrices")
@btime begin
    solve_slm_system(in_inc["Mdes"], vec(in_inc["rhs"]), in_inc["Mreg"],
                     vec(in_inc["rhsreg"]), in_inc["finalRP"], in_inc["Meq"], vec(in_inc["rhseq"]),
                     in_inc["Mineq"], vec(in_inc["rhsineq"]); use_sparse = true)
end
=#

# Run tests
@testset "Coefficients of least-square spline" begin
    for in_data in [in_inc, in_dec, in_sin, in_sinnoineq]
        @test maximum(abs.(in_data["coef"] -
                           solve_slm_system(in_data["Mdes"], vec(in_data["rhs"]), in_data["Mreg"],
                                            vec(in_data["rhsreg"]), in_data["finalRP"], in_data["Meq"], vec(in_data["rhseq"]),
                                            in_data["Mineq"], vec(in_data["rhsineq"])))) < 5e-3
    end

    # Check no constraints doesn't error
    @test noconstraints["coefs"] ≈ solve_slm_system(in_sinnoineq["Mdes"], vec(in_sinnoineq["rhs"]), in_sinnoineq["Mreg"],
                                                    vec(in_sinnoineq["rhsreg"]), in_sinnoineq["finalRP"],
                                                    Matrix{Float64}(undef, 0, 0), Vector{Float64}(undef, 0),
                                                    in_sinnoineq["Mineq"], vec(in_sinnoineq["rhsineq"]))
end
