module Elbo
using DiffResults, SpecialFunctions, ForwardDiff, StatsFuns, LinearAlgebra, Distributions
import ..HAFVFunivariate
import ..HAFVFmultivariate
import ..get_params
export ELBOentropy
include("univariate_normal.jl")
include("multivariate_normal.jl")
include("entropy.jl")
end
