function LB_nigentropy(mu,k,a,b)
Elogsig = log(b)-digamma(a)
return 0.5 + 0.918938533204673 + 0.5 * Elogsig - 0.5* log(k)    +LB_igentropy(a,b)
end
function LB_igentropy(a,b)
a+log(b)+lgamma(a)-(1.0+a)*digamma(a)
end
function LB_betaentropy(a,b)
lbeta(a,b)-(a-1.0)*digamma(a)-(b-1.0)*digamma(b)+(a+b-2.0)*digamma(a+b)
end


function ELBOentropy(s::HAFVFunivariate)
    L = LB_nigentropy(get_params(s.z)...)
    L += LB_betaentropy(get_params(s.w)...)
    L += LB_betaentropy(get_params(s.b)...)
end
function ELBOentropy(s::HAFVFmultivariate)
    L = LB_niwentropy(get_params(s.z)...)
    L += LB_betaentropy(get_params(s.w)...)
    L += LB_betaentropy(get_params(s.b)...)
end
function LB_niwentropy(μ,κ,η,Λ)
    n = size(μ,1)
    entropy_InverseWishart(Λ,η)+.5(ElogIW(Λ,η)-n*log(κ))
end
function entropy_InverseWishart(Ψ, df) #A::Distributions.InverseWishart) # gupta 2013
	p = size(Ψ,1)
	return StatsFuns.logmvgamma(p,df/2) + p*df/2 + (p+1)/2 * logdet(Ψ/2) - (p+df+1)/2 * logmvdigamma(p,df)
end
function logmvdigamma(d::Int64,q)
	sum(i->digamma((q-d+i)/2),1:d)
end

