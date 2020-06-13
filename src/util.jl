# Transferring parameters into NamedTuple and SLArray to avoid passing around
# unnecessary information about the parameters.
function parameters_to_named_tuple(pvec::AbstractVector{S}) where {S <: AbstractParameter}
    tuple_names = Tuple(p.key  for p in pvec)
    tuple_vals  = map(p -> p.value, pvec)
    return NamedTuple{tuple_names}(tuple_vals)
end

"""
```
detexify(s::String)

detexify(s::Symbol)
```

Remove Unicode characters from the string `s`, replacing them with ASCII
equivalents. For example, `detexify(\"π\")` returns `\"pi\"`.

This code is copied directly from DSGE.jl.
"""
function detexify(s::String)
    s = replace(s, "α" => "alpha")
    s = replace(s, "β" => "beta")
    s = replace(s, "γ" => "gamma")
    s = replace(s, "δ" => "delta")
    s = replace(s, "ϵ" => "epsilon")
    s = replace(s, "ε" => "epsilon")
    s = replace(s, "ζ" => "zeta")
    s = replace(s, "η" => "eta")
    s = replace(s, "θ" => "theta")
    s = replace(s, "ι" => "iota")
    s = replace(s, "κ" => "kappa")
    s = replace(s, "λ" => "lambda")
    s = replace(s, "μ" => "mu")
    s = replace(s, "ν" => "nu")
    s = replace(s, "ξ" => "xi")
    s = replace(s, "π" => "pi")
    s = replace(s, "ρ" => "rho")
    s = replace(s, "ϱ" => "rho")
    s = replace(s, "σ" => "sigma")
    s = replace(s, "ς" => "sigma")
    s = replace(s, "τ" => "tau")
    s = replace(s, "υ" => "upsilon")
    s = replace(s, "ϕ" => "phi")
    s = replace(s, "φ" => "phi")
    s = replace(s, "χ" => "chi")
    s = replace(s, "ψ" => "psi")
    s = replace(s, "ω" => "omega")

    s = replace(s, "Α" => "Alpha")
    s = replace(s, "Β" => "Beta")
    s = replace(s, "Γ" => "Gamma")
    s = replace(s, "Δ" => "Delta")
    s = replace(s, "Ε" => "Epsilon")
    s = replace(s, "Ζ" => "Zeta")
    s = replace(s, "Η" => "Eta")
    s = replace(s, "Θ" => "Theta")
    s = replace(s, "Ι" => "Iota")
    s = replace(s, "Κ" => "Kappa")
    s = replace(s, "Λ" => "Lambda")
    s = replace(s, "Μ" => "Mu")
    s = replace(s, "Ν" => "Nu")
    s = replace(s, "Ξ" => "Xi")
    s = replace(s, "Π" => "Pi")
    s = replace(s, "Ρ" => "Rho")
    s = replace(s, "Σ" => "Sigma")
    s = replace(s, "Τ" => "Tau")
    s = replace(s, "Υ" => "Upsilon")
    s = replace(s, "Φ" => "Phi")
    s = replace(s, "Χ" => "Chi")
    s = replace(s, "Ψ" => "Psi")
    s = replace(s, "Ω" => "Omega")

    s = replace(s, "∂" => "d") # This line is not in the original DSGE.detexify though

    s = replace(s, "′" => "'")

    return s
end

function detexify(s::Symbol)
    Symbol(detexify(string(s)))
end

function mat_to_jld2(fn::String, outfn::String)
    mat_data = matread(fn)
    JLD2.jldopen(outfn, true, true, true, IOStream) do file
        for (k, v) in mat_data
            write(file, k, v)
        end
    end
end

"""
```
sparse_accumarray(subs, val, sz=(maximum(subs),))
```

constructs a sparse array using the same algorithm as
`accumarray`. See either `VectorizedRoutines.jl` or
MATLAB's documentation for more details
"""
function sparse_accumarray(subs, val, sz=(maximum(subs),))
    A = spzeros(eltype(val), sz...)
    for i = 1:length(val)
        @inbounds A[subs[i]] += val[i]
    end
    A
end
