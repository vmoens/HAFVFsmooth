struct NormalInverseGammaDistribution
	μ::A
	κ::A
	α::A
	β::A
end
struct NormalInverseWishartDistribution
	μ::A
	κ::A
	η::A
	Λ::A
end
struct BetaDistribution
	α::A
	β::A
end
mutable struct HAFVFunivariate
	ApproxPost::NormalInverseGammaDistribution
	PrevPost::NormalInverseGammaDistribution
	InitPrior::NormalInverseGammaDistribution

	wApproxPost::BetaDistribution
	wPrevPost::BetaDistribution
	wInitPrior::BetaDistribution

	bApproxPost::BetaDistribution
	bPrevPost::BetaDistribution
	bInitPrior::BetaDistribution

	function HAFVFunivariate(;μ=0.0,κ=1.0,α=1.0,β=1.0,
				 ϕᵅ,ϕᵝ,
				 βᵅ,βᵝ)

		new()
	end
end
mutable struct HAFVFmultivariate
	ApproxPost::NormalInverseWishartDistribution
	PrevPost::NormalInverseWishartDistribution
	InitPrior::NormalInverseWishartDistribution

	wApproxPost::BetaDistribution
	wPrevPost::BetaDistribution
	wInitPrior::BetaDistribution

	bApproxPost::BetaDistribution
	bPrevPost::BetaDistribution
	bInitPrior::BetaDistribution

end

