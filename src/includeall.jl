using DifferentialEquations, LabelledArrays, ModelConstructors, NLsolve, OrderedCollections
using Random, Roots
using EconPDEs: StateGrid

import DiffEqBase: initialize!

# src directory
include("abstract_NLCT_model.jl")
include("util.jl")

# auxiliary/
include("auxiliary/initialize_grid.jl")
include("auxiliary/investment.jl")

# models/li2020
include("models/li2020/li2020.jl")
include("models/li2020/augment_endogenous_variables.jl")
include("models/li2020/eqcond.jl")
include("models/li2020/initialize.jl")
include("models/li2020/subspecs.jl")
