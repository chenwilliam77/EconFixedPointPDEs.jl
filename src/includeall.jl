# using DifferentialEquations,
using FastGaussQuadrature, FileIO, ForwardDiff, Interpolations, JLD2, LinearAlgebra
using ModelConstructors, NLsolve, NLPModelsIpopt, OrderedCollections, OrdinaryDiffEq, Printf, Random, Roots
using SparseArrays, StaticArrays, StatsBase, UnPack, VectorizedRoutines.Matlab

using EconPDEs: StateGrid
using NLPModels: ADNLSModel, LLSModel , FeasibilityFormNLS#, @lencheck, AbstractNLSModel, NLPModelMeta, NLSMeta, NLSCounters
# import NLPModels: hess_structure!, jac_structure!, hess_coord!, jac_coord!, jac_structure_residual!, hess_structure_residual!,
#     FeasibilityFormNLS
# ADD STATICARRAYS TO REPLACE THE VECTORS ON WHICH WE ARE ITERATING, ALSO TO MAKE NOJUMP ODE FASTER, also LuxurySparse
# for sparse equivalent

import Base: eltype, getindex
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


# Copy implementation of LLSMatrixModel, which has hess_structure! implemented.
# However, it appears not to work properly
#=function LLSMatrixModel(A :: AbstractMatrix, b :: AbstractVector;
                        x0 :: AbstractVector = zeros(eltype(A), size(A,2)),
                        lvar :: AbstractVector = fill(eltype(A)(-Inf), size(A, 2)),
                        uvar :: AbstractVector = fill(eltype(A)(Inf), size(A, 2)),
                        C :: AbstractMatrix  = Matrix{eltype(A)}(undef, 0, 0),
                        lcon :: AbstractVector = eltype(A)[],
                        ucon :: AbstractVector = eltype(A)[],
                        y0 :: AbstractVector = zeros(eltype(A), size(C,1)),
                        name :: String = "generic-LLSModel"
                       )
  nequ, nvar = size(A)
  ncon = size(C, 1)
  nnzjF = issparse(A) ? nnz(A) : nequ * nvar
  nnzh  = issparse(A) ? nnz(A' * A) : nvar * nvar
  nnzj  = issparse(C) ? nnz(C) : ncon * nvar
  meta = NLPModelMeta(nvar, x0=x0, lvar=lvar, uvar=uvar, ncon=ncon, y0=y0, lin=1:ncon,
                      nln=Int[], lcon=lcon, ucon=ucon, nnzj=nnzj, nnzh=nnzh, name=name)
  nls_meta = NLSMeta(nequ, nvar, nnzj=nnzjF, nnzh=0)

  LLSMatrixModel(meta, nls_meta, NLSCounters(), A, b, C)
end

function hess_structure!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzh rows cols
  AtA = tril(nls.A' * nls.A)
  if issparse(AtA)
    I, J, V = findnz(AtA)
    rows .= I
    cols .= J
  else
    n = size(nls.A)
    I = ((i,j) for i = 1:n, j = 1:n if i ≥ j)
    rows .= getindex.(I, 1)
    cols .= getindex.(I, 2)
  end
  return rows, cols
end

function hess_coord!(nls :: LLSMatrixModel, x :: AbstractVector, vals :: AbstractVector; obj_weight = 1.0)
  @lencheck nls.meta.nvar x
  @lencheck nls.nls_meta.nnzh vals
  increment!(nls, :neval_hess)
  AtA = tril(nls.A' * nls.A)
  if issparse(AtA)
    vals .= AtA.nzval
  else
    vals .= (AtA[i,j] for i = 1:n, j = 1:n if i ≥ j)
  end
  return vals
end

function jac_structure!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.meta.nnzj rows cols
  if issparse(nls.C)
    I, J, V = findnz(nls.C)
    rows .= I
    cols .= J
  else
    m, n = size(nls.C)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
  end
  return rows, cols
end

function jac_coord!(nls :: LLSMatrixModel, x :: AbstractVector, vals :: AbstractVector)
  @lencheck nls.meta.nvar x
  @lencheck nls.meta.nnzj vals
  increment!(nls, :neval_jac)
  if issparse(nls.C)
    vals .= nls.C.nzval
  else
    vals .= nls.C[:]
  end
  return vals
end

function jac_structure_residual!(nls :: LLSMatrixModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck nls.nls_meta.nnzj rows
  @lencheck nls.nls_meta.nnzj cols
  if issparse(nls.A)
    I, J, V = findnz(nls.A)
    rows .= I
    cols .= J
  else
    m, n = size(nls.A)
    I = ((i,j) for i = 1:m, j = 1:n)
    rows .= getindex.(I, 1)[:]
    cols .= getindex.(I, 2)[:]
  end
  return rows, cols
end

function hess_structure_residual!(nls :: AbstractLLSModel, rows :: AbstractVector{<: Integer}, cols :: AbstractVector{<: Integer})
  @lencheck 0 rows
  @lencheck 0 cols
  return rows, cols
end
=#
