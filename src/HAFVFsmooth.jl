module HAFVFsmooth
using DiffResults, ForwardDiff, StatsFuns, SpecialFunctions
include("types.jl")
include("smooth.jl")
include("elbo/elbo.jl")
using elbo
include("entropy.jl")
end # module
