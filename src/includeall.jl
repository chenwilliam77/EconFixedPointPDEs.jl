# using DifferentialEquations,
using BandedMatrices, DiffEqOperators, FastGaussQuadrature, FileIO, ForwardDiff, Interpolations, JLD2, LinearAlgebra
using ModelConstructors, NLsolve, NLPModelsIpopt, OrderedCollections, OrdinaryDiffEq, Printf, Random, Roots
using SparseArrays, StatsBase, UnPack, VectorizedRoutines.Matlab

using EconPDEs: StateGrid, implicit_timestep, finiteschemesolve
using NLPModels: ADNLSModel, LLSModel , FeasibilityFormNLS

import Base: eltype, getindex
import DiffEqBase: initialize!, solve
import Base.show

function Base.show(io::IO,  ::MIME"text/plain", sg::StateGrid)
    @printf io "%s-element %s\n" string(length(sg)) string(typeof(sg))
end

# src/ directory
include("abstract_NLCT_model.jl")
include("util.jl")

# auxiliary/
include("auxiliary/initialize_grid.jl")
include("auxiliary/investment.jl")

# solve/
include("solve/solve.jl")
include("solve/pseudo_transient_relaxation.jl")
include("solve/differentiate.jl")
include("solve/init_derivatives.jl")
include("solve/util.jl")

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

# models/brusan
include("models/brusan/brusan.jl")
include("models/brusan/augment_variables.jl")
include("models/brusan/eqcond.jl")
include("models/brusan/initialize.jl")
include("models/brusan/subspecs.jl")
