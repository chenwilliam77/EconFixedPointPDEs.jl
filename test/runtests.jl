using SafeTestsets

include("util.jl")

# solve/
include("solve/solve_nojump.jl")

# models/
include("models/li2020.jl")
include("models/nojump_eqm.jl")

# const GROUP = get(ENV, "GROUP", "All")
# const is_APPVEYOR = Sys.iswindows() && haskey(ENV,"APPVEYOR")
# const is_TRAVIS = haskey(ENV,"TRAVIS")

# Start Test Script

# @time begin
#     if GROUP == "All" || GROUP == "Interface"
#         # @time @safetestset "Basic SDO Examples" begin include("BasicSDOExamples.jl") end
#     end
# end
