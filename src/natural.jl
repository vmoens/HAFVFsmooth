struct naturalNormalInverseGammaDistribution
	η1
	η2
	η3
	η4
end
function natural(z::NormalInverseGammaDistribution)
	m = z.μ
	k = z.κ
	a = z.α
	b = z.β
        km = k*m
	return naturalNormalInverseGammaDistribution(a-.5,-b - .5*(km*m),km,-k/2)
end
function base(z::naturalNormalInverseGammaDistribution)
	k = -2 * z.η4
	m = z.η3 / k
	a = z.η1 + 0.5
	b = -z.η2 + - 0.5 * (z.η3*m)
	NormalInverseGammaDistribution(m,k,a,b)
end

import Base: +,-
for op in (:+,:-)
@eval function $op(a::naturalNormalInverseGammaDistribution,b::naturalNormalInverseGammaDistribution)
	naturalNormalInverseGammaDistribution(a.η1+b.η1,
					a.η2+b.η2,
					a.η3+b.η3,
					a.η4+b.η4)
end
end
