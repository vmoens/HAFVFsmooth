module HAFVFsmooth
using DiffResults, ForwardDiff, StatsFuns, SpecialFunctions, Distributions
export HAFVFunivariate, HAFVFmultivariate
include("types.jl")

include("elbo/elbo.jl")
using .Elbo

include("smooth.jl")
end # module
