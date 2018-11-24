module HAFVFsmooth
using DiffResults, ForwardDiff, StatsFuns, SpecialFunctions
include("types.jl")
include("smooth.jl")
include("elbo.jl")
include("entropy.jl")
end # module
