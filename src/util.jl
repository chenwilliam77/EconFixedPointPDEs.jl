function parameters_to_named_tuple(pvec::AbstractVector{S}) where {S <: AbstractParameter}
    tuple_names = Tuple(p.key  for p in pvec)
    tuple_vals  = map(p -> p.value, pvec)
    return NamedTuple{tuple_names}(tuple_vals)
end
