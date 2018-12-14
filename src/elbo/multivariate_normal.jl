
function ∇ELBO!(D, x, s::HAFVFmultivariate{:ForwardDiff}, stm1::HAFVFmultivariate{:ForwardDiff}, s0::HAFVFmultivariate{:ForwardDiff}, lag = 1)
    @assert s0.γ == one(s0.γ)
    out = ∇ELBOniw_forwarddiff!(x, 
                    get_params(s,        (s0.γ != 1))..., 
                    get_params(stm1,     (s0.γ != 1))..., 
                    get_params(s0,       (s0.γ != 1))..., lag)
    D .= out[2]
    out[1]
end
function ∇ELBO!(D, x, s::HAFVFmultivariate{:Flux}, stm1::HAFVFmultivariate{:Flux}, s0::HAFVFmultivariate{:Flux}, lag = 1)
    @assert s0.γ == one(s0.γ)
    out = ∇ELBOniw_flux!(x, s, stm1, s0, lag)
    D .= out[2]
    out[1]
end



∇Elbo = DiffResults.GradientResult(zeros(4))
function make_∇ELBOniw()
    F(x,μ, κ, η, Λ,
        μ1,κ1,η1,Λ1,αᵅ1,βᵅ1,αᵝ1,βᵝ1,
        μ2,κ2,η2,Λ2,αᵅ2,βᵅ2,αᵝ2,βᵝ2, t) = a->ELBOniw(x,
                        μ, κ, η, Λ, a..., μ1,κ1,η1,Λ1,αᵅ1,βᵅ1,αᵝ1,βᵝ1,μ2,κ2,η2,Λ2,αᵅ2,βᵅ2,αᵝ2,βᵝ2,t)
    function ∇ELBOniw!(x, μ, κ, η, Λ, αᵅ, βᵅ, αᵝ, βᵝ,
              μ1, κ1, η1, Λ1, αᵅ1, βᵅ1, αᵝ1, βᵝ1,
    		  μ2, κ2, η2, Λ2, αᵅ2, βᵅ2, αᵝ2, βᵝ2,
              t)
              ForwardDiff.gradient!(∇Elbo,
                     F(x,μ, κ, η, Λ,μ1,κ1,η1,Λ1,αᵅ1,βᵅ1,αᵝ1,βᵝ1,μ2,κ2,η2,Λ2,αᵅ2,βᵅ2,αᵝ2,βᵝ2,t),
                     [αᵅ;βᵅ;αᵝ;βᵝ])
              ∇Elbo.value, ∇Elbo.derivs[1]
    end
end
∇ELBOniw_forwarddiff! = make_∇ELBOniw()

function ∇ELBOniw_flux!(x, s, stm1, s0, lag)
    @assert lag == 1
    d = param([get_params(s.w)...;get_params(s.b)...])
    L = ELBOniw(x, get_params(s.z)..., d...,
                get_params(stm1,     (s0.γ != 1))...,
                get_params(s0,       (s0.γ != 1))...)
    Tracker.back!(L)
    return L.data, d.grad
end

function ELBOniw(r::AbstractArray{T,1},
		 μ, κ, η, Λ, αᵅ, βᵅ, αᵝ, βᵝ,
		μ1,κ1,η1,Λ1,αᵅ1,βᵅ1,αᵝ1,βᵝ1,
		μ2,κ2,η2,Λ2,αᵅ2,βᵅ2,αᵝ2,βᵝ2::Real,
		t=1) where T

	if t==one(t)
		Eα = αᵅ/(αᵅ+βᵅ)
	else
		Eα = Εα(αᵅ,βᵅ,t)
	end
	Eβ = αᵝ/(αᵝ+βᵝ)

	L = -.5ElogIW(Λ,η)-log2π*length(r)/2-.5 * η * tr(Λ\((μ-r)*(μ-r)' + Λ/(η*κ)))
	Ltm1 = LB_niwPart(μ, κ, η, Λ, μ1,κ1,η1,Λ1)
	L0 =   LB_niwPart(μ, κ, η, Λ, μ2,κ2,η2,Λ2)
	L+=Eα * (Ltm1-L0)+L0
	L-=LogPartition_NIW(αᵅ, βᵅ,μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2,t)

        Ltm1 = LB_betaPart(αᵅ,βᵅ,αᵅ1,βᵅ1)
        L0   = LB_betaPart(αᵅ,βᵅ,αᵅ2,βᵅ2)
        L+=Eβ*(Ltm1-L0)+L0
        L-=LogPartition_Beta(αᵝ,βᵝ,αᵅ1,βᵅ1,αᵅ2,βᵅ2)

        L+=LB_betaPart(αᵝ,βᵝ,αᵝ1,βᵝ1)

        L   
end

function natural_param_NIW(μ,κ,η,Λ)
p = length(μ)
n1a = (0.5 * (-2-p-η))
n2a = -κ * (μ*μ')/2 - Λ/2
n3a = κ*μ
n4a = -κ/2
return n1a,n2a,n3a,n4a
end
function LogPartition_NIW(aa,ba,μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2,t)
        # Log partition function and Taylor series
	if t==one(t)
        	Eb=aa/(aa+ba)
	        Ebm1=1.0-Eb
        	Varbeta=Eb*Ebm1/(1.0+aa+ba)
	else
 		Eb,Varbeta = EαVα(aa,ba,t)
		Ebm1=1.0-Eb
	end
	p=length(μ1)
    (n1a,n2a,n3a,n4a),(n1b,n2b,n3b,n4b) = map(natural_param_NIW,[μ1,μ2],[κ1,κ2],[η1;η2],[Λ1,Λ2])
#        n1a,n1b = (v->(1/2 * (-2-p-v))).([η1;η2])
#        n2a,n2b = ((m,k,Λ)->-k * (m*m')/2 - Λ/2).([μ1,μ2],[κ1,κ2],[Λ1,Λ2])
#        n3a,n3b = ((m,k)->k*m).([μ1,μ2],[κ1,κ2])
#        n4a,n4b = -[κ1,κ2]/2

        lb = -Eb * Aniw(n1a,n2a,n3a,n4a) -
                        (1-Eb)*Aniw(n1b,n2b,n3b,n4b)
        A,hA = F_hessAniw(Eb,n1a,n2a,n3a,n4a,
                              n1b,n2b,n3b,n4b)
        lb += A + hA*Varbeta/2
        return lb
end

function LB_betaPart(a,b,a0,b0)
        dab = digamma(a+b)
	ab = (a,b)
Eloga,Elogb = @. digamma(ab)-dab
return (a0-1.0)* Eloga+ (b0-1.0) * Elogb - lbeta(a0,b0)
end

function LB_niwPart(μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2) # gupta 2013
        p = length(μ1)
    return -.5(ElogIW(Λ1,η1) - p * log(κ2)) - p*log2π/2 - .5 * κ2 * η1 * ( tr(Λ1 \ ((μ1-μ2)*(μ1-μ2)' + Λ1 / (κ1 * η1) ))) + 
 		η2/2 * (logdet(Λ2) - p * log(2)) - StatsFuns.logmvgamma(p,η2/2) - .5 * (η2+p+1) * ElogIW(Λ1,η1) - .5*η1 * tr(Λ1\Λ2 )
end


function LogPartition_Beta(aa,ba,a1,b1,a2,b2)
Eb=aa/(aa+ba)
Ebm1=1.0-Eb
Vb=Eb*Ebm1/(1.0+aa+ba)
lb=0.0
lb-=(Eb * lbeta(a1,b1) + Ebm1 * lbeta(a2,b2))
lb+=Tayl(Eb,Ebm1,Vb,a1,a2)
lb+=Tayl(Eb,Ebm1,Vb,b1,b2)
lb-=Tayl(Eb,Ebm1,Vb,a1+b1,a2+b2)
return lb
end
function ElogIW(Λ,η) 
        p = size(Λ,1)
        -p * (log2π-logπ) + logdet(Λ) - logmvdigamma(p,η)
end
function logmvdigamma(d::Int64,q)
	sum(i->digamma((q-d+i)/2),1:d)
end
function Aniw(n1,n2,n3,n4)
    p = length(n3)
	v = -2n1-p-2
	k = -2n4
	logn4 = log(k)
	logdetn2n3n4 = logdet(-2n2 + n3*n3' / (2n4))
        return .5(v * p * log(2)+
	  	    log2π*p-
		    p * logn4-v*logdetn2n3n4+
		    2logmvgamma(p,v/2))
end
∇₂Aniw_Eb!(H,Eb,μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2) = ForwardDiff.hessian!(H,
             Eb->Aniw(Eb[1]*μ1+(1-Eb[1])*μ2,Eb[1]*κ1+(1-Eb[1])*κ2,Eb[1]*η1+(1-Eb[1])*η2,Eb[1]*Λ1+(1-Eb[1])*Λ2),
             [Eb])
@generated function F_hessAniw(Eb::T,μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2) where T
    quote
        H = $(DiffResults.HessianResult([one(T)]))
        ∇₂Aniw_Eb!(H,Eb,μ1,κ1,η1,Λ1,μ2,κ2,η2,Λ2)
        return DiffResults.value(H),DiffResults.hessian(H)[1]
    end
end
function Tayl(Eb,Vb,δ::Array{T,1}...) where T
Δ = sum(Eb.*[δ...])
∇h = hcat(δ...)'*hκDirich(Δ)*hcat(δ...)
h = κDirich(Δ)
h+tr(Vb*∇h)/2
#return ∇h.value+tr(Vb*DiffResults.hessian(∇h))/2
end
function Tayl(Eb,Ebm1,Vb,a1::Real,a2::Real)
v1=Eb*a1+Ebm1*a2
return lgamma(v1) + 0.5 * ((a1-a2)^2.0)*polygamma(1,v1)*Vb
end

function hκDirich(δ)
pgδ = @. polygamma(1,δ)
pgδ2 = polygamma(1,sum(δ))
diagm(pgδ)-pgδ2
end
function κDirich(δ)
sum(lgamma,δ)-lgamma(sum(δ))
end

