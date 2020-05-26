"""
```
```
The AbstractNLCTModel is defined as a subtype of AbstractModel to accommodate
the numerical methods and procedures specific to global solutions of
nonlinear continuous-time models.
"""
abstract type AbstractNLCTModel{T} <: AbstractModel{T} end
