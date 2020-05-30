"""
```
abstract type AbstractNLCTModel{T} <: AbstractModel{T} end
```

The AbstractNLCTModel is defined as a subtype of AbstractModel to accommodate
the numerical methods and procedures specific to global solutions of
nonlinear continuous-time models.
"""
abstract type AbstractNLCTModel{T} <: AbstractModel{T} end

function Base.show(io::IO, m::AbstractNLCTModel)
    @printf io "Model \n"
    @printf io "description:\n %s\n"          description(m)
end

get_type(m::AbstractNLCTModel{T}) where {T <: Real} = T

### Auxiliary access functions for typical things in an AbstractNLCTModel
get_settings(m::AbstractNLCTModel) = m.settings
get_endogenous_variables(m::AbstractNLCTModel) = m.endogenous_variables
get_exogenous_shocks(m::AbstractNLCTModel) = m.exogenous_shocks
get_observables(m::AbstractNLCTModel) = m.observables
get_stategrid(m::AbstractNLCTModel) = m.stategrid
get_parameters(m::AbstractNLCTModel) = m.parameters
get_pseudo_observables(m::AbstractNLCTModel) = m.pseudo_observables
get_test_settings(m::AbstractNLCTModel) = m.test_settings

n_endogenous_variables(m::AbstractNLCTModel) = length(get_endogenous_variables(m))
n_exogenous_shocks(m::AbstractNLCTModel) = length(get_exogenous_shocks(m))


"""
```
set_boundary_conditions(m::AbstractNLCTModel, k::Symbol, v::AbstractArray{S}) where {S <: Real}
```

set boundary conditions for an instance of an `AbstractNLCTModel` for endogenous variables
characterized by differential equations.
"""
function set_boundary_conditions!(m::AbstractNLCTModel, k::Symbol, v::AbstractArray{S}) where {S <: Real}
    bc = get_setting(m, :boundary_conditions)
    bc[k] = v
end
