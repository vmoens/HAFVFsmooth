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


function ELBOentropy(μτ,κτ,ατ,βτ,αᵅ,βᵅ,αᵝ,βᵝ)
        L = LB_nigentropy(μτ,κτ,ατ,βτ)
        L += LB_betaentropy(αᵅ,βᵅ)
        L += LB_betaentropy(αᵝ,βᵝ)
end
