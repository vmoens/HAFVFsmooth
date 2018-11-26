abstract type HAFVFtypes end
struct NormalInverseGammaDistribution{A} <: HAFVFtypes
	μ::A
	κ::A
	α::A
	β::A
end
struct NormalInverseWishartDistribution{A,B,C} <: HAFVFtypes
	μ::A
	κ::B
	η::B
	Λ::C
end
struct BetaDistribution{A} <: HAFVFtypes
	α::A
	β::A
end

mutable struct HAFVFunivariate <: HAFVFtypes
	z::NormalInverseGammaDistribution

	w::BetaDistribution

	b::BetaDistribution
	
	γ::Real

	function HAFVFunivariate(μ,κ,α,β,
				 ϕᵅ,ϕᵝ,
				 βᵅ,βᵝ,
				 γ=0.999)

		new(NormalInverseGammaDistribution(μ,κ,α,β),
		    BetaDistribution(ϕᵅ,ϕᵝ),
		    BetaDistribution(βᵅ,βᵝ),
		    γ
		    )
	end
end
mutable struct HAFVFmultivariate <: HAFVFtypes
	z::NormalInverseWishartDistribution

	w::BetaDistribution

	b::BetaDistribution

	γ::Real

	function HAFVFmultivariate(μ,κ,η,Λ,
				 ϕᵅ,ϕᵝ,
				 βᵅ,βᵝ,
				 γ=1.0)

		new(NormalInverseWishartDistribution(μ,κ,η,Λ),
		    BetaDistribution(ϕᵅ,ϕᵝ),
		    BetaDistribution(βᵅ,βᵝ),
		    γ
		    )
	end
end



function get_params(s::T) where T<:HAFVFtypes
    g = tuplejoin(get_params.(map(f->getfield(s, f),fieldnames(T)))...)
    return g
end
function get_params(s::T, get_γ::Bool) where T<:Union{HAFVFmultivariate,HAFVFunivariate}
    get_params(s)[1:end-1+get_γ]
end
function get_params(s)
    return s
end
tuplejoin(t1::Tuple, t2::Tuple, t3...) = tuplejoin((t1..., t2...), t3...)
tuplejoin(t1::Tuple, t2, t3...) = tuplejoin((t1..., t2), t3...)
tuplejoin(t::Tuple) = t
tuplejoin(x...) = x
