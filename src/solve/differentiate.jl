function differentiate(stategrid, funcvars, operators, derivatives)

# Differentiate

for (k, v) in funcvars
    for state in keys(stategrid.x)
	    derivatives[Symbol(:∂, k, :∂, state)] = operators[:A] * v
    end
end
