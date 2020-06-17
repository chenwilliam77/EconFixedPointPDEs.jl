# This script tests the internal functions of eqcond.jl
using Test, OrderedCollections
using BenchmarkTools
include("../../../src/includeall.jl")

rp = "../../reference/models/li2020"

inside_input = load(joinpath(rp, "inside_iteration_input.jld2"))
Q̂_input = load(joinpath(rp, "Qhat_calculation_input.jld2"))
Q̂_out   = load(joinpath(rp, "hat_Q_calculation_output.jld2"))

# Q̂ should either be initialized as zero, or as the previous value
# to speed up convergence. It shouldn't be the input from Q̂_input!
stategrid = StateGrid(OrderedDict(:w => vec(Q̂_input["w_grid"])))
#=
@btime begin
    Q̂_calculation(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
              vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
              Q̂_input["Qval"], Q̂_input["lambda"])
end
=#
my_Q̂ = Q̂_calculation(stategrid, zeros(length(stategrid)), vec(Q̂_input["muw_vec"]), vec(Q̂_input["sigmaw_vec"]),
              vec(Q̂_input["kappaw_vec"]), vec(Q̂_input["rf_vec"]), vec(Q̂_input["rg_vec"]), vec(Q̂_input["rh_vec"]),
              Q̂_input["Qval"], Q̂_input["lambda"])
@test maximum(abs.(my_Q̂ - vec(Q̂_out["hat_Q_vec"]))) < 5e-4
