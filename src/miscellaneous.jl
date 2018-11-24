function iCovBeta(ατ,βτ)
        poly_a = polygamma(1,ατ)
        poly_b = polygamma(1,βτ)
        poly_ab = polygamma(1,ατ+βτ)

        A = poly_a-poly_ab
        C = poly_b-poly_ab
        B = -poly_ab
    
        Λα = [C -B;-B A]/(-B*B+A*C)

end


function Abeta(Eβ,αᵅtm1,βᵅtm1,αᵅ₀,βᵅ₀)
        αᵅτ = Eβ*(αᵅtm1-αᵅ₀)+αᵅ₀
        βᵅτ = Eβ*(βᵅtm1-βᵅ₀)+βᵅ₀
	lgamma(αᵅτ)+lgamma(βᵅτ)-lgamma(βᵅτ+αᵅτ)
end
    function ∇2Beta(Eβ, αᵅtm1, βᵅtm1, αᵅ₀, βᵅ₀) # /Users/OldVince/Dropbox/Julia/AFVF_Examples/../AFVFtools/../NDiff/NDiff.jl, line 307:
            _tmp4 = βᵅtm1 - βᵅ₀
            _tmp3 = Eβ * _tmp4 + βᵅ₀
            _tmp2 = αᵅtm1 - αᵅ₀
            _tmp7 = _tmp2 + _tmp4
            _tmp5 = Eβ * _tmp2 + αᵅ₀
            D2 = -((_tmp7 * trigamma(_tmp3 + _tmp5)) * _tmp7) + _tmp4 ^ 2 * trigamma(_tmp3) + _tmp2 ^ 2 * trigamma(_tmp5)
    end

function Anig(Eα,μτm1,κτm1,ατm1,βτm1,μτ0,κτ0,ατ0,βτ0)
	ατp = Eα*(ατm1-ατ0)+ατ0
	κτp = Eα*(κτm1-κτ0)+κτ0
	μτp = (Eα*κτm1*μτm1+(1-Eα)*κτ0*μτ0)/κτp
	βτp = Eα*(βτm1-βτ0)+βτ0+.5(Eα*(1-Eα)*κτm1*κτ0*(μτm1-μτ0)^2)
	Anig(μτp,κτp,ατp,βτp)
end
function Anig(μτp,κτp,ατp,βτp)
	lgamma(ατp)-ατp*log(βτp)-1/2*log(κτp)
end
function ∇2NIG(Eα, μτm1, κτm1, ατm1, βτm1, μτ0, κτ0, ατ0, βτ0) # /Users/OldVince/Dropbox/Julia/AFVF_Examples/../AFVFtools/../NDiff/NDiff.jl, line 307:
        _tmp18 = κτm1 - κτ0
        _tmp13 = ατm1 - ατ0
        _tmp12 = Eα * _tmp13 + ατ0
        _tmp11 = βτm1 - βτ0
        _tmp10 = 1 - Eα
        _tmp9 = 0.5_tmp10
        _tmp6 = (μτm1 - μτ0) ^ 2 * κτ0 * κτm1
        _tmp19 = 0.5_tmp6
        _tmp5 = Eα * _tmp6
        _tmp8 = -(0.5_tmp5) + _tmp11 + _tmp6 * _tmp9
        _tmp16 = _tmp11 + (-Eα + _tmp10) * _tmp19
        _tmp15 = 1 / (Eα * _tmp11 + _tmp5 * _tmp9 + βτ0)
        _tmp14 = _tmp13 * _tmp15
        D2 = -(((-(_tmp15 ^ 2) * _tmp16 * _tmp8 + -(2 * _tmp15 * _tmp19)) * _tmp12 + _tmp14 * _tmp16 + _tmp14 * _tmp8)) + -(-((1 / (Eα * _tmp18 + κτ0)) ^ 2) * 0.5 * _tmp18 ^ 2) + _tmp13 ^ 2 * trigamma(_tmp12)
end

function ELBOentropy(αᴿ,βᴿ,αᵅ,βᵅ,αᵝ,βᵝ)
        L = LB_betaentropy(αᴿ,βᴿ)
        L += LB_betaentropy(αᵅ,βᵅ)
        L += LB_betaentropy(αᵝ,βᵝ)
end

function LB_normPart_generic(Ex,Vx,Emu,Vmu,Elogsig,Esigm1)
return -.5Elogsig-0.918938533204673-.5((Ex-Emu)^2.+Vx+Vmu)*Esigm1
end
function LB_nigPart(mu,k,a,b,mu0,k0,a0,b0)
Emu = mu
Esigm1 = a/b
Varmu = b/(a*k)
Elogsig = log(b)-digamma(a)
return -.5(Elogsig-log(k0))-0.918938533204673-.5k0*((Emu-mu0)^2. + Varmu)*Esigm1 +
        -lgamma(a0) + a0*log(b0) - (a0+1)*Elogsig - b0 * Esigm1
end
function LB_betaPart(a,b,a0,b0)
        dab = digamma(a+b)
	ab = (a,b)
Eloga,Elogb = @. digamma(ab)-dab
return (a0-1.0)* Eloga+ (b0-1.0) * Elogb - lbeta(a0,b0)
end
