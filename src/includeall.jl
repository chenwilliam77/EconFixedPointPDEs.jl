using Dierckx, DifferentialEquations, FastGaussQuadrature, FileIO, ForwardDiff, Interpolations, JLD2, LinearAlgebra
using ModelConstructors, NLsolve, NLPModelsIpopt, OrderedCollections, Printf, Random, Roots
using SparseArrays, VectorizedRoutines.Matlab

using EconPDEs: StateGrid
using JSOSolvers: tron
using NLPModels: LLSModel

import Base: getindex
import DiffEqBase: initialize!, solve

# src/ directory
include("abstract_NLCT_model.jl")
include("util.jl")

# auxiliary/
include("auxiliary/initialize_grid.jl")
include("auxiliary/investment.jl")

# solve/
include("solve/solve.jl")

# SLM/
include("SLM/slm.jl")
include("SLM/default_slm_kwargs.jl")
include("SLM/eval.jl")
include("SLM/property.jl")
include("SLM/solve_slm_system.jl")
include("SLM/util.jl")

# models/li2020
include("models/li2020/li2020.jl")
include("models/li2020/augment_variables.jl")
include("models/li2020/eqcond.jl")
include("models/li2020/initialize.jl")
include("models/li2020/subspecs.jl")
