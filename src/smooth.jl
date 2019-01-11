include("smooth_settings.jl")
(s::HAFVFmultivariate)(x) = __smooth__(s,x)
(s::HAFVFunivariate)(x) = __smooth__(s,x)

function __smooth__(s,x)
    s0 = deepcopy(s)
    approx_post = [deepcopy(s)]
    elbo = zeros(size(x))
    for (i,x) in enumerate(x)
        elbo[i] = optimise(x, s, approx_post[end], s0)
	    push!(approx_post,deepcopy(s))
    end
    return elbo, approx_post
end

function optimise(x, s::HAFVFtypes, stm1::HAFVFtypes, s0::HAFVFtypes,gamma=0.999, lag=1)
    ELBOo = -Inf

    l = 8

    i = 1
    ELBO = updParams!(x,s,stm1,s0,lag,i)

    while true
        i+=1
        ELBOn = ELBO
        dE = (ELBOn-ELBOo)/ELBOo
        ELBOo = ELBOn
        if abs(dE)<THRESHOLD || i>=1000
            if i==ITERATION_FAILURE && abs(dE)>1e-2
                print("failed\t")
                ELBO-=1000.0
            end
            break
        else
        ELBO = updParams!(x,s,stm1,s0,lag,i)
        end
    end
    ELBO

end



# Normal Update
function make_updParams()
    ∇αβ = zeros(4)
    Λ   = zeros(2,2)
    Df  = DiffResults.GradientResult

    function update_wb!(x,s,stm1,s0,lag=1)
        gam = s.γ
        
        elbo = Elbo.∇ELBO!(∇αβ, x, s, stm1, s0, lag)
        
        w = s.w
        b = s.b
        iCovBeta!(Λ, get_params(w)...)
        w_tmp = BetaDistribution( (Λ*∇αβ[1:2] .+ 1)... )
        iCovBeta!(Λ, get_params(b)...)
        b_tmp = BetaDistribution( (Λ*∇αβ[3:4] .+ 1)... )
        
        c = DAMP_RATIO
        s.w = damp(w_tmp, w, c)
        s.b = damp(b_tmp, b, c)

        elbo
    end
    function damp(w_tmp::BetaDistribution, w::BetaDistribution, c=DAMP_RATIO)
        return BetaDistribution((w_tmp.α ^ c) * (w.α ^ (1-c)),
                                (w_tmp.β ^ c) * (w.β ^ (1-c)))
    end
    function updParams!(x,
                        s,stm1,s0,
                        lag=1,i=1)
    	
        elbo = update_wb!(x,s,stm1,s0,lag)

        update_z!(x,s,stm1,s0,lag)
        
        elbo += ELBOentropy(s)
    end
end
updParams! = make_updParams()


function update_z!(x,s::HAFVFunivariate,stm1::HAFVFunivariate,s0::HAFVFunivariate,lag=1)
    μτ,κτ,ατ,βτ,ϕᵅτ,ϕᵝτ,βᵅτ,βᵝτ = get_params(s)
    μτm1,κτm1,ατm1,βτm1,ϕᵅm1,ϕᵝm1,βᵅm1,βᵝm1 = get_params(stm1)
    μ0,κ0,α0,β0,ϕᵅ0,ϕᵝ0,βᵅ0,βᵝ0 = get_params(s0)
    
    if lag == one(lag)
    	Ea = Εα(ϕᵅτ,ϕᵝτ)
    else
    	Ea = Εα(ϕᵅτ,ϕᵝτ,lag)
    end
    κτ = Ea * κτm1 + (1-Ea) * κ0 + 1.0
    ατ = Ea * ατm1 + (1-Ea) * α0 + 0.5
    μτ = (Ea * κτm1 * μτm1 + (1-Ea) * κ0 * μ0 + x)/κτ
    βτ = Ea * βτm1 + (1-Ea) * β0 + 0.5(Ea*κτm1*(μτm1-μτ)^2 + (1-Ea)*κ0*(μ0-μτ)^2 + (x-μτ)^2)
    
    s.z = NormalInverseGammaDistribution(μτ, κτ, ατ, βτ)
end

function update_z!(x,s::HAFVFmultivariate,stm1::HAFVFmultivariate,s0::HAFVFmultivariate,lag=1)
    μτ,   κτ,   ητ,   Λτ,   ϕᵅτ,    ϕᵝτ,    βᵅτ,    βᵝτ = get_params(s)
    μτm1, κτm1, ητm1, Λτm1, ϕᵅm1,   ϕᵝm1,   βᵅm1,   βᵝm1 = get_params(stm1)
    μ0,   κ0,   η0,   Λ0,   ϕᵅ0,    ϕᵝ0,    βᵅ0,    βᵝ0 = get_params(s0)
    
    if lag == one(lag)
    	Ea = Εα(ϕᵅτ,ϕᵝτ)
    else
    	Ea = Εα(ϕᵅτ,ϕᵝτ,lag)
    end
    κτ = Ea * κτm1 + (1-Ea) * κ0 + 1.0
    ητ = Ea * ητm1 + (1-Ea) * η0 + 1.0
    μτ = (Ea .* κτm1 .* μτm1 .+ (1-Ea) .* κ0 .* μ0 .+ x) ./ κτ
    Λτ = Ea .* Λτm1 .+ (1-Ea) .* Λ0 .+ (Ea .* κτm1 .* (μτm1-μτ)*(μτm1-μτ)' .+ (1-Ea) .* κ0 .* (μ0-μτ)*(μ0-μτ)' .+ (x-μτ)*(x-μτ)')
    
    s.z = NormalInverseWishartDistribution(μτ, κτ, ητ, Λτ)
end

# inverse covariance for beta
@inbounds function iCovBeta!(Λ,ατ,βτ)
        poly_a = polygamma(1,ατ)
        poly_b = polygamma(1,βτ)
        poly_ab = polygamma(1,ατ+βτ)

        A = poly_a-poly_ab
        C = poly_b-poly_ab
        B = -poly_ab
    
	Λ[1,1] = C
	Λ[2,1] = -B
	Λ[1,2] = -B
	Λ[2,2] = A
	broadcast!(x->x/(-B*B+A*C),Λ,Λ)
	Λ
end
function Εα(aa,ba)
	aa/(aa+ba)
end
function Εα(aa,ba,lag)
	Eb = exp(lgamma(ba)+lgamma(aa+lag)-lbeta(aa,ba)-lgamma(aa+ba+lag))
end












