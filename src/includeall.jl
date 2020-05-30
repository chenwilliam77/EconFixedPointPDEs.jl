using DifferentialEquations, LabelledArrays, ModelConstructors, NLsolve, OrderedCollections, Printf
using Random, Roots
using EconPDEs: StateGrid

import DiffEqBase: initialize!, solve

# src directory
include("abstract_NLCT_model.jl")
include("util.jl")

# auxiliary/
include("auxiliary/initialize_grid.jl")
include("auxiliary/investment.jl")

# solve/
include("solve/solve.jl")

# models/li2020
include("models/li2020/li2020.jl")
include("models/li2020/augment_variables.jl")
include("models/li2020/eqcond.jl")
include("models/li2020/initialize.jl")
include("models/li2020/subspecs.jl")
