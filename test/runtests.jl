using HAFVFsmooth
using LinearAlgebra, Flux


print("Testing univariate\n")
x = cumsum(randn(100))
s = HAFVFsmooth.HAFVFunivariate(1.0 .+ rand(8)...)
S = s(x)
@time S = s(x)

print("Testing multivariate\n")
np = 4
X = randn(np,100)
X = cumsum([X[:,k] for k in 1:100])
s = HAFVFmultivariate(randn(np),5.0,5.0,Matrix{Float64}(I, np,np), rand(4) .+ 1.0 ...)
S = s(X)
@time S = s(X)

