function differentiate(stategrid, diffvars, operators, derivatives)

# Differentiate

for (k, v) in diffvars
    for state in keys(stategrid.x)
	    derivatives[Symbol(:∂, k, :∂, state)] = operators[:A] * v
    end
end
