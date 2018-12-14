module HAFVFsmooth
using DiffResults, ForwardDiff, StatsFuns, SpecialFunctions, Distributions, Flux
using Flux: Tracker
export HAFVFunivariate, HAFVFmultivariate
include("types.jl")

include("elbo/elbo.jl")
using .Elbo

include("smooth.jl")
end # module
