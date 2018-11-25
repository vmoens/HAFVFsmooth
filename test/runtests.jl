using HAFVFsmooth
x = cumsum(randn(100))
s = HAFVFsmooth.HAFVFunivariate(1.0 .+ rand(8)...)

S = s(x)

