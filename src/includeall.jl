using Dierckx, DifferentialEquations, FastGaussQuadrature, FileIO, ForwardDiff, Interpolations, JLD2, LinearAlgebra
using ModelConstructors, NLsolve, NLPModelsIpopt, OrderedCollections, Printf, Random, Roots
using SparseArrays, StatsBase, VectorizedRoutines.Matlab

using EconPDEs: StateGrid
using JSOSolvers: tron
using NLPModels# : ADNLSModel, LLSModel, @lencheck, jac_residual, jac_op

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

function NLPModels.hess(model::LLSModel, x::AbstractVector; obj_weight::Real=one(eltype(x)))
    @lencheck model.meta.nvar x
    J = jac_residual(model, x)
    return obj_weight * (J' * J)
end

function NLPModels.hess_op(model::LLSModel, x::AbstractVector; obj_weight::Real=one(eltype(x)))
    @lencheck model.meta.nvar x
    J = jac_op(model, x)
    return obj_weight * J' * J
end
