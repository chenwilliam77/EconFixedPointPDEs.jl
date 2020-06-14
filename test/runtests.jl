using SafeTestsets

include("util.jl")

# solve/
include("solve/solve_nojump.jl")

# models/
include("models/li2020.jl")
include("models/nojump_eqm.jl")

# SLM/
include("SLM/default_slm_kwargs.jl")
include("SLM/util.jl")
include("SLM/property.jl")
include("SLM/solve_slm_system.jl")
include("SLM/slm.jl")
include("SLM/eval.jl")

# const GROUP = get(ENV, "GROUP", "All")
# const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
# const is_TRAVIS = haskey(ENV,"TRAVIS")

# Start Test Script

# @time begin
#     if GROUP == "All" || GROUP == "Interface"
#         # @time @safetestset "Basic SDO Examples" begin include("BasicSDOExamples.jl") end
#     end
# end
