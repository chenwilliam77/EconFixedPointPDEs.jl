
"""
```
average_update(new::AbstractArray{S}, old::AbstractArray{S}, learning_rate::S) where {S <: Real}
```

updates `old` with `new` by taking a weighted average according to `learning_rate`.
"""
@inline function average_update(new::AbstractArray{S}, old::AbstractArray{S}, learning_rate::S) where {S <: Real}
    return learning_rate .* new + (1 - learning_rate) .* old
end

"""
```
function calculate_func_error(new_funcvar::OrderedDict, old_funcvar::OrderedDict, error_method::Symbol)

function calculate_func_error(new_funcvar::AbstractArray{S}, old_funcvar::AbstractArray{S}, error_method::Symbol) where {S <: Real}
```

calculates the error between the proposal and the current guesses
for the functional variables.
"""
@inline function calculate_func_error(new_funcvar::OrderedDict, old_funcvar::OrderedDict, error_method::Symbol)

    func_error = if error_method == :total_error
        sum([sum(abs.(v1 - v2)) for (v1, v2)
             in zip(values(new_funcvar), values(old_funcvar))])
    elseif error_method in [:L∞, :Linf, :max_abs_error, :maximum_absolute_error]
        maximum([maximum(abs.(v1 - v2)) for (v1, v2)
                 in zip(values(new_funcvar), values(old_funcvar))])
    elseif error_method in [:L², :L2, :squared_error]
        sum([sum((v1 - v2) .^ 2) for (v1, v2)
             in zip(values(new_funcvar), values(old_funcvar))])
    end

    return func_error
end

@inline function calculate_func_error(new_funcvar::AbstractArray{S}, old_funcvar::AbstractArray{S}, error_method::Symbol) where {S <: Real}

    func_error = if error_method == :total_error
        sum(abs.(new_funcvar - old_funcvar))
    elseif error_method in [:L∞, :Linf, :max_abs_error, :maximum_absolute_error]
        maximum(abs.(new_funcvar - old_funcvar))
    elseif error_method in [:L², :L2, :squared_error]
        sum((new_funcvar - old_funcvar) .^ 2)
    end

    return func_error
end
