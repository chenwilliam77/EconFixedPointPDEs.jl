# Extend later to multiple dimensions by using multple dispatch, probably we'll now use a Vector of Tuples of Ints
# or by using generated functions
function init_derivatives!(m::AbstractNLCTModel, instructions::AbstractDict{Symbol, Vector{Int}},
                           statevars::Vector{Symbol} = collect(keys(get_state_variables(m))))

    derivs = get_derivatives(m)
    state = statevars[1]

    n = length(derivs) # Get existing length of derivatives that have already been added
    for (k, v) in instructions
        for i in v
            n += 1
            if i == 1
                derivs[Symbol(:∂, k, :_∂, state)] = n
            elseif i == 2
                derivs[Symbol(:∂², k, :_∂, state, "²")] = n
            end
        end
    end
end

function init_derivatives!(m::AbstractNLCTModel, instructions::AbstractDict{Symbol, Vector{Tuple{Int, Int}}},
                           statevars::Vector{Symbol} = collect(keys(get_state_variables(m))))

    derivs = get_derivatives(m)
    state1 = statevars[1]
    state2 = statevars[2]

    n = length(derivs) # Get existing length of derivatives that have already been added
    for (k, v) in instructions
        for i in v
            n += 1
            if i == (1, 0)
                derivs[Symbol(:∂, k, :_∂, state1)] = n
            elseif i == (2, 0)
                derivs[Symbol(:∂², k, :_∂, state1, "²")] = n
            elseif i == (0, 1)
                derivs[Symbol(:∂, k, :_∂, state2)] = n
            elseif i == (0, 2)
                derivs[Symbol(:∂², k, :_∂, state2, "²")] = n
            elseif i == (1, 1)
                derivs[Symbol(:∂², k, :_∂, state1, :∂, state2)] = n
            end
        end
    end
end


"""
```
function standard_derivs(dims::Int)
```

returns instructions for which derivatives
to calculate for standard continuous time models,
depending on the dimension `dims` of the state space.

For a one-dimensional model, we request the first and second derivatives.

For a multi-dimensional model, we request the first, second,
and (first) mixed partial derivatives, e.g.
for a 2D model, we want ∂f_∂x, ∂f²_∂x2, ∂f²_∂x∂y, ∂f_∂y, ∂f²_∂y2.

The instructions are returned as a Vector of Ints or Vector of Tuples of Ints.
"""
function standard_derivs(dims::Int)
    if dims == 1
        v = Vector{Int}(undef, 2)
        v[1] = 1
        v[2] = 2
        return v
    elseif dims == 2
        v = Vector{Tuple{Int, Int}}(undef, 5)
        v[1] = (1, 0)
        v[2] = (0, 1)
        v[3] = (2, 0)
        v[4] = (1, 1)
        v[5] = (0, 2)
        return v
    end
end
