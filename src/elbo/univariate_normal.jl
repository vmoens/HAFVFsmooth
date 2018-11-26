######## No gamma (3 levels) ########
#===================================#

function ∇ELBO!(D, x, s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate, lag)
    gam = s0.γ
    if gam != one(gam)
    	if lag == one(lag)
            ∇ELBO_gam!(D, x, s, stm1, s0)
            elbo   = ELBO_gam_compact(x, s, stm1, s0) 
    	else
    		error("Both lag and upper fixed decay not implemented yet")
    	end
    else
    	if lag == one(lag)
            elbo   = ∇ELBO!(D, x, s, stm1, s0)
    	else
            D = DiffResults.gradient(
                           ForwardDiff.gradient!(∇Elbo,wb->ELBO(x, s, stm1, s0, wb=wb), get_wb(s)))
    		elbo   = Df.value
    	end
    end
    elbo
end


function ELBO(x,s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate, lag)
    get_γ = (s0.γ != 1)
    if lag == one(lag)
        return ELBO(x, 
                    get_params(s, get_γ)..., 
                    get_params(stm1, get_γ)..., 
                    get_params(s0, get_γ)...)
    else
        return ELBO(x, 
                    get_params(s, get_γ)..., 
                    get_params(stm1, get_γ)..., 
                    get_params(s0, get_γ)..., lag)
    end
end
function ELBO(x,
	      μτ,κτ,ατ,βτ,
              αᵅ,βᵅ,
              αᵝ,βᵝ,
              μτm1,κτm1,ατm1,βτm1,
              αᵅtm1,βᵅtm1,
              αᵝtm1,βᵝtm1,
              μτ0,κτ0,ατ0,βτ0,
              αᵅ₀,βᵅ₀,
              αᵝ₀,βᵝ₀,
		lag)

        L = LB_normPart_generic(x,0.,μτ,βτ/(ατ*κτ),log(βτ)-digamma(ατ),ατ/βτ)
        Ltm1 = LB_nigPart(μτ,κτ,ατ,βτ,
                          μτm1,κτm1,ατm1,βτm1)
        L0   = LB_nigPart(μτ,κτ,ατ,βτ,
                          μτ0,κτ0,ατ0,βτ0)
	Eα = Εα(αᵅ,βᵅ,lag)
        L+=Eα*(Ltm1-L0)+L0
        L-=LogPartition_NIG(αᵅ,βᵅ,μτm1,κτm1,ατm1,βτm1,μτ0,κτ0,ατ0,βτ0,lag)

	Eβ = Εα(αᵝ,βᵝ,lag)
        Ltm1 = LB_betaPart(αᵅ,βᵅ,αᵅtm1,βᵅtm1)
        L0   = LB_betaPart(αᵅ,βᵅ,αᵅ₀,βᵅ₀)
        L+=Eβ*(Ltm1-L0)+L0
        L-=LogPartition_Beta(αᵝ,βᵝ,αᵅtm1,βᵅtm1,αᵅ₀,βᵅ₀,lag)

        L+=LB_betaPart(αᵝ,βᵝ,αᵝtm1,βᵝtm1)

        L
end
# Normal case
function ELBO(x,
	      μτ,κτ,ατ,βτ,
              αᵅ,βᵅ,
              αᵝ,βᵝ,
              μτm1,κτm1,ατm1,βτm1,
              αᵅtm1,βᵅtm1,
              αᵝtm1,βᵝtm1,
              μτ0,κτ0,ατ0,βτ0,
              αᵅ₀,βᵅ₀,
              αᵝ₀,βᵝ₀)

        L = LB_normPart_generic(x,0.,μτ,βτ/(ατ*κτ),log(βτ)-digamma(ατ),ατ/βτ)
        Eα = αᵅ/(αᵅ+βᵅ)
	ατp = Eα*(ατm1-ατ0)+ατ0
	κτp = Eα*(κτm1-κτ0)+κτ0
	μτp = (Eα*κτm1*μτm1+(1-Eα)*κτ0*μτ0)/κτp
	βτp = Eα*(βτm1-βτ0)+βτ0+.5(Eα*(1-Eα)*κτm1*κτ0*(μτm1-μτ0)^2)
        L += LB_nigPart(μτ,κτ,ατ,βτ,
                          μτp,κτp,ατp,βτp)
	L-=Eα*(1-Eα)/(αᵅ+βᵅ+1) * ∇2NIG(Eα, μτm1, κτm1, ατm1, βτm1, μτ0, κτ0, ατ0, βτ0)/2

        Eβ = αᵝ/(αᵝ+βᵝ)
	αᵅτ = Eβ*(αᵅtm1-αᵅ₀)+αᵅ₀
	βᵅτ = Eβ*(βᵅtm1-βᵅ₀)+βᵅ₀
        L += LB_betaPart(αᵅ,βᵅ,αᵅτ,βᵅτ)
	L -= Eβ*(1-Eβ)/(1+αᵝ+βᵝ)*∇2Beta(Eβ, αᵅtm1, βᵅtm1, αᵅ₀, βᵅ₀)/2

        #L+=LB_betaPart(αᵝ,βᵝ,αᵝ₀,βᵝ₀)
        L+=LB_betaPart(αᵝ,βᵝ,αᵝtm1,βᵝtm1)

        L
end
∇ELBO!(D, x, s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate) = HAFVFunivariate(D, x, get_params(s)[1:end-1]..., get_params(stm1)[1:end-1]..., get_params(s0)[1:end-1]...)
function ∇ELBO!(D1, x, μτ, κτ, ατ, βτ, αᵅ, βᵅ, αᵝ, βᵝ, μτm1, κτm1, ατm1, βτm1, αᵅtm1, βᵅtm1, αᵝtm1, βᵝtm1, μτ0, κτ0, ατ0, βτ0, αᵅ₀, βᵅ₀, αᵝ₀, βᵝ₀) 
        begin  
            _tmp125 = βᵝtm1 - 1.0
            _tmp124 = αᵝtm1 - 1.0
            _tmp114 = (1 / (ατ * κτ)) * βτ
            _tmp111 = (1 / βτ) * ατ
            _tmp216 = 0.5_tmp111
            _tmp109 = log(βτ) - digamma(ατ)
            _tmp92 = βτm1 - βτ0
            _tmp76 = ατm1 - ατ0
            _tmp75 = _tmp76 ^ 2
            _tmp74 = κτm1 - κτ0
            _tmp104 = 0.5 * _tmp74 ^ 2
            _tmp71 = κτm1 * μτm1
            _tmp70 = αᵅ + βᵅ
            _tmp175 = 1 / _tmp70
            _tmp192 = _tmp175 * αᵅ
            _tmp205 = _tmp192 * _tmp92 + βτ0
            _tmp204 = 1 - _tmp192
            _tmp218 = 0.5_tmp204
            _tmp203 = _tmp192 * _tmp74 + κτ0
            _tmp217 = 1 / _tmp203
            _tmp231 = -(_tmp217 ^ 2)
            _tmp202 = _tmp192 * _tmp76 + ατ0
            _tmp236 = -(_tmp104 * _tmp231) + _tmp75 * trigamma(_tmp202)
            _tmp155 = digamma(_tmp70)
            _tmp188 = -(trigamma(_tmp70))
            _tmp108 = digamma(βᵅ) - _tmp155
            _tmp106 = digamma(αᵅ) - _tmp155
            _tmp156 = -(_tmp175 * _tmp192)
            _tmp158 = -_tmp156
            _tmp145 = _tmp156 * _tmp76
            _tmp94 = _tmp156 * _tmp92
            _tmp86 = _tmp156 + _tmp175
            _tmp157 = -_tmp86
            _tmp136 = _tmp76 * _tmp86
            _tmp95 = _tmp86 * _tmp92
            _tmp170 = digamma(_tmp202)
            _tmp142 = _tmp217 * _tmp74
            _tmp88 = _tmp204 * _tmp86
            _tmp87 = _tmp157 * _tmp192 + _tmp88
            _tmp85 = _tmp156 * _tmp204 + _tmp158 * _tmp192
            _tmp66 = κτ0 * μτ0
            _tmp232 = (_tmp192 * _tmp71 + _tmp204 * _tmp66) * _tmp217
            _tmp237 = μτ - _tmp232
            _tmp238 = _tmp114 + _tmp237 ^ 2.0
            _tmp63 = αᵅtm1 - αᵅ₀
            _tmp146 = _tmp106 * _tmp63
            _tmp62 = _tmp63 ^ 2
            _tmp61 = αᵝ + βᵝ
            _tmp194 = 1 / _tmp61
            _tmp206 = _tmp194 * αᵝ
            _tmp166 = digamma(_tmp61)
            _tmp190 = -(trigamma(_tmp61))
            _tmp207 = _tmp206 * _tmp63 + αᵅ₀
            _tmp122 = 1 - _tmp206
            _tmp160 = -(_tmp194 * _tmp206)
            _tmp147 = _tmp160 + _tmp194
            _tmp105 = _tmp207 - 1.0
            _tmp57 = βᵅtm1 - βᵅ₀
            _tmp148 = _tmp108 * _tmp57
            _tmp80 = _tmp57 + _tmp63
            _tmp226 = _tmp80 ^ 2
            _tmp58 = _tmp206 * _tmp57 + βᵅ₀
            _tmp107 = _tmp58 - 1.0
            _tmp56 = _tmp57 ^ 2
            _tmp54 = _tmp56 * _tmp57 * polygamma(2, _tmp58)
            _tmp53 = _tmp62 * _tmp63 * polygamma(2, _tmp207)
            _tmp141 = _tmp238 * _tmp74
            _tmp186 = 2.0 * _tmp203 * _tmp237 ^ 1.0
            _tmp47 = 2 * _tmp217 ^ 1 * _tmp74
            _tmp41 = _tmp75 * _tmp76 * polygamma(2, _tmp202)
            _tmp179 = 1 / (1 + _tmp61)
            _tmp209 = 0.5_tmp179
            _tmp196 = _tmp122 * _tmp206 * _tmp209
            _tmp151 = -(_tmp179 * _tmp196)
            _tmp180 = 1 / (1 + _tmp70)
            _tmp197 = _tmp180 * _tmp192 * _tmp204
            _tmp135 = -(_tmp180 * _tmp197)
            _tmp140 = -(_tmp217 * _tmp232) * _tmp74
            _tmp32 = (μτm1 - μτ0) ^ 2 * κτ0 * κτm1
            _tmp229 = _tmp218 * _tmp32
            _tmp234 = _tmp229 + _tmp92
            _tmp212 = _tmp192 * _tmp32
            _tmp230 = _tmp205 + _tmp212 * _tmp218
            _tmp235 = 1 / _tmp230
            _tmp223 = 0.5_tmp212
            _tmp103 = 2_tmp32
            _tmp181 = 0.5_tmp103
            _tmp89 = 0.5_tmp32
            _tmp222 = (-_tmp192 + _tmp204) * _tmp89 + _tmp92
            _tmp199 = _tmp207 + _tmp58
            _tmp233 = -(_tmp226 * trigamma(_tmp199)) + _tmp56 * trigamma(_tmp58) + _tmp62 * trigamma(_tmp207)
            _tmp161 = digamma(_tmp199)
            _tmp153 = _tmp57 * (digamma(_tmp58) - _tmp161) + _tmp63 * (digamma(_tmp207) - _tmp161)
            _tmp174 = _tmp226 * _tmp80 * polygamma(2, _tmp199)
            _tmp144 = -(_tmp156 * _tmp89) + _tmp158 * _tmp89
            _tmp143 = _tmp85 * _tmp89 + _tmp94
            _tmp132 = -(_tmp86 * _tmp89) + _tmp157 * _tmp89
            _tmp128 = _tmp87 * _tmp89 + _tmp95
            _tmp38 = -_tmp223 + _tmp234
            _tmp44 = -(_tmp192 * _tmp89) + _tmp234
            _tmp51 = _tmp156 * _tmp229 + _tmp158 * _tmp223 + _tmp94
            _tmp43 = _tmp157 * _tmp223 + _tmp88 * _tmp89 + _tmp95
            _tmp118 = _tmp235 * _tmp76
            _tmp165 = log(_tmp230)
            _tmp127 = _tmp202 * _tmp235
            _tmp99 = 1 / (_tmp192 * _tmp229 + _tmp205)
            _tmp191 = -(_tmp99 ^ 2)
            _tmp201 = _tmp181 * _tmp191
            _tmp102 = _tmp76 * _tmp99
            _tmp185 = _tmp191 * _tmp222
            _tmp97 = _tmp191 * _tmp76
            _tmp52 = _tmp51 * _tmp97
            _tmp45 = _tmp43 * _tmp97
            _tmp42 = 2 * _tmp191 * _tmp99 ^ 1
            _tmp183 = -(0.5 * _tmp103 * _tmp99) + _tmp191 * _tmp222 * _tmp44
            _tmp133 = _tmp183 * _tmp76
            _tmp48 = -((_tmp102 * _tmp222 + _tmp102 * _tmp44 + _tmp183 * _tmp202)) + _tmp236
            _tmp1 = _tmp102 * _tmp181
            v = (((((((((-0.5 * (_tmp109 - log(_tmp203)) - 0.918938533204673) - _tmp203 * _tmp216 * _tmp238) + -(lgamma(_tmp202)) + _tmp165 * _tmp202) - (1 + _tmp202) * _tmp109) - _tmp111 * _tmp230) + ((-0.5_tmp109 - 0.918938533204673) - ((x - μτ) ^ 2 + _tmp114) * _tmp216)) - (-(((-(0.5 * _tmp103 * _tmp235) + -(_tmp235 ^ 2) * _tmp222 * _tmp38) * _tmp202 + _tmp118 * _tmp222 + _tmp118 * _tmp38)) + _tmp236) * _tmp180 * _tmp192 * _tmp218) + ((_tmp105 * _tmp106 + _tmp107 * _tmp108) - lbeta(_tmp207, _tmp58))) - _tmp196 * _tmp233) + ((_tmp124 * (digamma(αᵝ) - _tmp166) + _tmp125 * (digamma(βᵝ) - _tmp166)) - lbeta(αᵝtm1, βᵝtm1))
            D1 .= begin  
                    reshape([-(((-((((-(_tmp42 * _tmp43) * _tmp222 + _tmp157 * _tmp201) * _tmp44 + -(_tmp201 * _tmp43) + _tmp132 * _tmp185) * _tmp202 + _tmp1 * _tmp157 + _tmp102 * _tmp132 + _tmp133 * _tmp86 + _tmp222 * _tmp45 + _tmp44 * _tmp45)) + -(-(_tmp231 * _tmp47 * _tmp86) * _tmp104) + _tmp41 * _tmp86) * _tmp197 + (_tmp135 + _tmp180 * _tmp87) * _tmp48) * 0.5) + -((-(((_tmp157 * _tmp66 + _tmp71 * _tmp86) * _tmp217 + _tmp140 * _tmp86)) * _tmp186 + _tmp141 * _tmp86) * _tmp216) + -(_tmp109 * _tmp136) + -(_tmp111 * _tmp128) + -(_tmp136 * _tmp170) + -(_tmp142 * _tmp86) * -0.5 + _tmp105 * (_tmp188 + trigamma(αᵅ)) + _tmp107 * _tmp188 + _tmp127 * _tmp128 + _tmp136 * _tmp165, -(((-((((-(_tmp42 * _tmp51) * _tmp222 + _tmp158 * _tmp201) * _tmp44 + -(_tmp201 * _tmp51) + _tmp144 * _tmp185) * _tmp202 + _tmp1 * _tmp158 + _tmp102 * _tmp144 + _tmp133 * _tmp156 + _tmp222 * _tmp52 + _tmp44 * _tmp52)) + -(-(_tmp156 * _tmp231 * _tmp47) * _tmp104) + _tmp156 * _tmp41) * _tmp197 + (_tmp135 + _tmp180 * _tmp85) * _tmp48) * 0.5) + -((-(((_tmp156 * _tmp71 + _tmp158 * _tmp66) * _tmp217 + _tmp140 * _tmp156)) * _tmp186 + _tmp141 * _tmp156) * _tmp216) + -(_tmp109 * _tmp145) + -(_tmp111 * _tmp143) + -(_tmp142 * _tmp156) * -0.5 + -(_tmp145 * _tmp170) + _tmp105 * _tmp188 + _tmp107 * (_tmp188 + trigamma(βᵅ)) + _tmp127 * _tmp143 + _tmp145 * _tmp165, -((((-_tmp147 * _tmp206 + _tmp122 * _tmp147) * _tmp209 + _tmp151) * _tmp233 + (-(_tmp147 * _tmp174) + _tmp147 * _tmp53 + _tmp147 * _tmp54) * _tmp196)) + -(_tmp147 * _tmp153) + _tmp124 * (_tmp190 + trigamma(αᵝ)) + _tmp125 * _tmp190 + _tmp146 * _tmp147 + _tmp147 * _tmp148, -((((-_tmp160 * _tmp206 + _tmp122 * _tmp160) * _tmp209 + _tmp151) * _tmp233 + (-(_tmp160 * _tmp174) + _tmp160 * _tmp53 + _tmp160 * _tmp54) * _tmp196)) + -(_tmp153 * _tmp160) + _tmp124 * _tmp190 + _tmp125 * (_tmp190 + trigamma(βᵝ)) + _tmp146 * _tmp160 + _tmp148 * _tmp160], (4,))
                end
        end 
        v
    end











######## Gamma (4 levels, last deterministic) ########
#====================================================#
ELBO_gam(x, s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate) = ELBO_gam(x, get_params(s)[1:end-1]..., get_params(stm1)[1:end-1]..., get_params(s0)...)
function ELBO_gam(x,
	      μτ,κτ,ατ,βτ,
              αᵅ,βᵅ,
              αᵝ,βᵝ,
              μτm1,κτm1,ατm1,βτm1,
              αᵅtm1,βᵅtm1,
              αᵝtm1,βᵝtm1,
              μτ0,κτ0,ατ0,βτ0,
              αᵅ₀,βᵅ₀,
              αᵝ₀,βᵝ₀,
	      gam)

        L = LB_normPart_generic(x,0.,μτ,βτ/(ατ*κτ),log(βτ)-digamma(ατ),ατ/βτ)
        Eα = αᵅ/(αᵅ+βᵅ)
	ατp = Eα*(ατm1-ατ0)+ατ0
	κτp = Eα*(κτm1-κτ0)+κτ0
	μτp = (Eα*κτm1*μτm1+(1-Eα)*κτ0*μτ0)/κτp
	βτp = Eα*(βτm1-βτ0)+βτ0+.5(Eα*(1-Eα)*κτm1*κτ0*(μτm1-μτ0)^2)
        L += LB_nigPart(μτ,κτ,ατ,βτ,
                          μτp,κτp,ατp,βτp)
	L-=Eα*(1-Eα)/(αᵅ+βᵅ+1) * ∇2NIG(Eα, μτm1, κτm1, ατm1, βτm1, μτ0, κτ0, ατ0, βτ0)/2

        Eβ = αᵝ/(αᵝ+βᵝ)
	αᵅτ = Eβ*(αᵅtm1-αᵅ₀)+αᵅ₀
	βᵅτ = Eβ*(βᵅtm1-βᵅ₀)+βᵅ₀
        L += LB_betaPart(αᵅ,βᵅ,αᵅτ,βᵅτ)
	L -= Eβ*(1-Eβ)/(1+αᵝ+βᵝ)*∇2Beta(Eβ, αᵅtm1, βᵅtm1, αᵅ₀, βᵅ₀)/2

        #L+=LB_betaPart(αᵝ,βᵝ,αᵝ₀,βᵝ₀)
	αᵝτ = gam*(αᵝtm1-αᵝ₀)+αᵝ₀
	βᵝτ = gam*(βᵝtm1-βᵝ₀)+βᵝ₀
        L += LB_betaPart(αᵝ,βᵝ,αᵝτ,βᵝτ)

        L
end

## Gradient when upper level has a decay
    ELBO_gam_compact(x, s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate) = 
        ELBO_gam_compact(x, get_params(s)[1:end-1]..., get_params(stm1)[1:end-1]..., get_params(s0)...)
    function ELBO_gam_compact(x, μτ, κτ, ατ, βτ, αᵅ, βᵅ, αᵝ, βᵝ, μτm1, κτm1, ατm1, βτm1, αᵅtm1, βᵅtm1, αᵝtm1, βᵝtm1, μτ0, κτ0, ατ0, βτ0, αᵅ₀, βᵅ₀, αᵝ₀, βᵝ₀, gam) 
            _tmp44 = gam * (βᵝtm1 - βᵝ₀) + βᵝ₀
            _tmp43 = gam * (αᵝtm1 - αᵝ₀) + αᵝ₀
            _tmp31 = (1 / (ατ * κτ)) * βτ
            _tmp29 = (1 / βτ) * ατ
            _tmp28 = 0.5_tmp29
            _tmp27 = κτm1 - κτ0
            _tmp25 = log(βτ) - digamma(ατ)
            _tmp24 = ατm1 - ατ0
            _tmp22 = βᵅtm1 - βᵅ₀
            _tmp20 = αᵝ + βᵝ
            _tmp47 = (1 / _tmp20) * αᵝ
            _tmp50 = _tmp22 * _tmp47 + βᵅ₀
            _tmp46 = digamma(_tmp20)
            _tmp18 = αᵅtm1 - αᵅ₀
            _tmp51 = _tmp18 * _tmp47 + αᵅ₀
            _tmp42 = _tmp18 + _tmp22
            _tmp15 = βτm1 - βτ0
            _tmp12 = αᵅ + βᵅ
            _tmp48 = (1 / _tmp12) * αᵅ
            _tmp55 = _tmp15 * _tmp48 + βτ0
            _tmp54 = 1 - _tmp48
            _tmp56 = 0.5_tmp54
            _tmp53 = _tmp24 * _tmp48 + ατ0
            _tmp52 = _tmp27 * _tmp48 + κτ0
            _tmp45 = digamma(_tmp12)
            _tmp30 = 1 / _tmp52
            _tmp7 = (μτm1 - μτ0) ^ 2 * κτ0 * κτm1
            _tmp49 = 0.5_tmp7
            _tmp6 = _tmp48 * _tmp7
            _tmp10 = -(0.5_tmp6) + _tmp15 + _tmp56 * _tmp7
            _tmp33 = (-_tmp48 + _tmp54) * _tmp49 + _tmp15
            _tmp34 = 1 / (_tmp55 + _tmp56 * _tmp6)
            _tmp35 = _tmp24 * _tmp34
            _tmp8 = _tmp48 * _tmp49 * _tmp54 + _tmp55
            v = (((((((((-0.5 * (_tmp25 - log(_tmp52)) - 0.918938533204673) - ((μτ - _tmp30 * (_tmp48 * κτm1 * μτm1 + _tmp54 * κτ0 * μτ0)) ^ 2.0 + _tmp31) * _tmp28 * _tmp52) + -(lgamma(_tmp53)) + _tmp53 * log(_tmp8)) - (1 + _tmp53) * _tmp25) - _tmp29 * _tmp8) + ((-0.5_tmp25 - 0.918938533204673) - ((x - μτ) ^ 2 + _tmp31) * _tmp28)) - (-(((-(2 * _tmp34 * _tmp49) + -(_tmp34 ^ 2) * _tmp10 * _tmp33) * _tmp53 + _tmp10 * _tmp35 + _tmp33 * _tmp35)) + -(-(_tmp30 ^ 2) * 0.5 * _tmp27 ^ 2) + _tmp24 ^ 2 * trigamma(_tmp53)) * (1 / (1 + _tmp12)) * _tmp48 * _tmp56) + (((_tmp50 - 1.0) * (digamma(βᵅ) - _tmp45) + (_tmp51 - 1.0) * (digamma(αᵅ) - _tmp45)) - lbeta(_tmp51, _tmp50))) - (-((_tmp42 * trigamma(_tmp50 + _tmp51)) * _tmp42) + _tmp18 ^ 2 * trigamma(_tmp51) + _tmp22 ^ 2 * trigamma(_tmp50)) * 0.5 * (1 - _tmp47) * (1 / (1 + _tmp20)) * _tmp47) + (((_tmp43 - 1.0) * (digamma(αᵝ) - _tmp46) + (_tmp44 - 1.0) * (digamma(βᵝ) - _tmp46)) - lbeta(_tmp43, _tmp44))
    end
    
    ∇ELBO_gam!(D, x, s::HAFVFunivariate, stm1::HAFVFunivariate, s0::HAFVFunivariate) = 
        ∇ELBO_gam!(D, x, get_params(s)[1:end-1]..., get_params(stm1)[1:end-1]..., get_params(s0)...)
   @inbounds function ∇ELBO_gam!(D, x, 
                                 μτ, κτ, ατ, βτ, αᵅ, βᵅ, αᵝ, βᵝ, 
                                 μτm1, κτm1, ατm1, βτm1, αᵅtm1, βᵅtm1, αᵝtm1, βᵝtm1, 
                                 μτ0, κτ0, ατ0, βτ0, αᵅ₀, βᵅ₀, αᵝ₀, βᵝ₀, gam) 
                _tmp141 = (gam * (βᵝtm1 - βᵝ₀) + βᵝ₀) - 1.0
                _tmp139 = (gam * (αᵝtm1 - αᵝ₀) + αᵝ₀) - 1.0
                _tmp126 = log(βτ) - digamma(ατ)
                _tmp88 = κτm1 * μτm1
                _tmp86 = βτm1 - βτ0
                _tmp85 = _tmp86 * αᵅ
                _tmp166 = (1 / βτ) * ατ
                _tmp75 = κτ0 * μτ0
                _tmp74 = ατm1 - ατ0
                _tmp113 = _tmp74 ^ 2
                _tmp72 = αᵅ + βᵅ
                _tmp178 = -(trigamma(_tmp72))
                _tmp147 = digamma(_tmp72)
                _tmp143 = digamma(βᵅ) - _tmp147
                _tmp142 = digamma(αᵅ) - _tmp147
                _tmp71 = 1 / _tmp72
                _tmp77 = _tmp71 * αᵅ
                _tmp76 = 1 - _tmp77
                _tmp182 = 0.5_tmp76
                _tmp149 = -(_tmp71 ^ 2)
                _tmp73 = _tmp149 * αᵅ
                _tmp167 = _tmp71 + _tmp73
                _tmp189 = -_tmp167
                _tmp154 = -_tmp73
                _tmp125 = _tmp167 * _tmp74
                _tmp69 = κτm1 - κτ0
                _tmp117 = 0.5 * _tmp69 ^ 2
                _tmp83 = _tmp167 * _tmp69
                _tmp68 = _tmp69 * αᵅ
                _tmp169 = _tmp149 * _tmp68
                _tmp67 = αᵅtm1 - αᵅ₀
                _tmp96 = _tmp67 ^ 2
                _tmp65 = αᵝ + βᵝ
                _tmp179 = -(trigamma(_tmp65))
                _tmp64 = 1 / _tmp65
                _tmp87 = _tmp64 * αᵝ
                _tmp99 = 1 - _tmp87
                _tmp168 = 0.5_tmp99
                _tmp151 = -(_tmp64 ^ 2)
                _tmp66 = _tmp151 * αᵝ
                _tmp94 = _tmp64 + _tmp66
                _tmp93 = _tmp67 * _tmp94
                _tmp62 = βᵅtm1 - βᵅ₀
                _tmp97 = _tmp62 ^ 2
                _tmp95 = _tmp62 * _tmp94
                _tmp92 = _tmp62 + _tmp67
                _tmp61 = _tmp62 * αᵝ
                _tmp59 = _tmp151 * _tmp61
                _tmp58 = _tmp66 * _tmp67
                _tmp53 = _tmp73 * _tmp74
                _tmp50 = 0.5_tmp166
                _tmp118 = 1 / (1 + _tmp72)
                _tmp82 = _tmp68 * _tmp71 + κτ0
                _tmp81 = 1 / _tmp82
                _tmp180 = -(_tmp81 ^ 2)
                _tmp130 = 2_tmp81
                _tmp138 = _tmp169 * _tmp180
                _tmp49 = _tmp180 * _tmp83
                _tmp114 = _tmp74 * _tmp77 + ατ0
                _tmp161 = digamma(_tmp114)
                _tmp127 = polygamma(2, _tmp114) * _tmp113
                _tmp100 = 1 / (1 + _tmp65)
                _tmp89 = _tmp61 * _tmp64 + βᵅ₀
                _tmp145 = polygamma(2, _tmp89) * _tmp97
                _tmp122 = _tmp89 - 1.0
                _tmp90 = _tmp67 * _tmp87 + αᵅ₀
                _tmp144 = polygamma(2, _tmp90) * _tmp96
                _tmp121 = _tmp90 - 1.0
                _tmp28 = (μτm1 - μτ0) ^ 2 * κτ0 * κτm1
                _tmp190 = _tmp182 * _tmp28
                _tmp183 = 0.5_tmp28
                _tmp172 = 2_tmp183
                _tmp27 = _tmp88 * αᵅ
                _tmp134 = _tmp27 * _tmp71 + _tmp75 * _tmp76
                _tmp133 = μτ - _tmp134 * _tmp81
                _tmp135 = (1 / (ατ * κτ)) * βτ + _tmp133 ^ 2.0
                _tmp177 = 2.0 * _tmp133 * _tmp82
                _tmp24 = _tmp28 * αᵅ
                _tmp173 = _tmp149 * _tmp24
                _tmp21 = _tmp89 + _tmp90
                _tmp153 = digamma(_tmp21)
                _tmp146 = (digamma(_tmp89) - _tmp153) * _tmp62 + (digamma(_tmp90) - _tmp153) * _tmp67
                _tmp91 = polygamma(2, _tmp21) * _tmp92
                _tmp31 = -(_tmp92 ^ 2 * trigamma(_tmp21)) + _tmp96 * trigamma(_tmp90) + _tmp97 * trigamma(_tmp89)
                _tmp184 = _tmp168 * _tmp31
                _tmp20 = -(_tmp100 ^ 2) * _tmp184
                _tmp128 = -(_tmp167 * _tmp183) + _tmp183 * _tmp189
                _tmp112 = (-_tmp77 + _tmp76) * _tmp183 + _tmp86
                _tmp14 = _tmp24 * _tmp71
                _tmp137 = -(0.5_tmp173) + _tmp154 * _tmp183
                _tmp136 = (0.5 * _tmp154 * _tmp28 * _tmp71 + _tmp149 * _tmp86 + _tmp149 * _tmp190) * αᵅ
                _tmp6 = 0.5_tmp14
                _tmp104 = -_tmp6 + _tmp190 + _tmp86
                _tmp54 = _tmp149 * _tmp85 + _tmp154 * _tmp6 + _tmp173 * _tmp182
                _tmp106 = _tmp14 * _tmp182 + _tmp71 * _tmp85 + βτ0
                _tmp174 = 1 / _tmp106
                _tmp188 = _tmp114 * _tmp174
                _tmp162 = log(_tmp106)
                _tmp181 = -(_tmp174 ^ 2)
                _tmp115 = _tmp174 * _tmp74
                _tmp176 = _tmp104 * _tmp181
                _tmp187 = _tmp172 * _tmp176
                _tmp107 = _tmp181 * _tmp74
                _tmp48 = _tmp104 * _tmp107
                _tmp46 = 2 * _tmp174 * _tmp181
                _tmp43 = _tmp107 * _tmp112
                _tmp9 = _tmp172 * _tmp181
                _tmp185 = -(_tmp172 * _tmp174) + _tmp112 * _tmp176
                _tmp35 = -((_tmp104 * _tmp115 + _tmp112 * _tmp115 + _tmp114 * _tmp185)) + -(_tmp117 * _tmp180) + _tmp113 * trigamma(_tmp114)
                _tmp186 = _tmp118 * _tmp35
                _tmp42 = -(_tmp118 ^ 2) * _tmp35
                _tmp25 = _tmp186 * _tmp77
                _tmp3 = _tmp115 * _tmp172
                _tmp165 = _tmp167 * _tmp86 + _tmp189 * _tmp6 + _tmp167 * _tmp190
		D[1] = (trigamma(αᵅ) + _tmp178) * _tmp121 + -(((((-((((-(_tmp165 * _tmp46) * _tmp104 + _tmp128 * _tmp181) * _tmp112 + -(_tmp165 * _tmp9) + _tmp187 * _tmp189) * _tmp114 + _tmp115 * _tmp128 + _tmp125 * _tmp185 + _tmp165 * _tmp43 + _tmp165 * _tmp48 + _tmp189 * _tmp3)) + -(-(_tmp130 * _tmp49) * _tmp117) + _tmp125 * _tmp127) * _tmp118 + _tmp42) * _tmp77 + _tmp167 * _tmp186) * _tmp76 + _tmp189 * _tmp25) * 0.5) + -((-(((_tmp167 * _tmp88 + _tmp189 * _tmp75) * _tmp81 + _tmp134 * _tmp49)) * _tmp177 + _tmp135 * _tmp83) * _tmp50) + -(_tmp125 * _tmp126) + -(_tmp125 * _tmp161) + -(_tmp165 * _tmp166) + -(_tmp81 * _tmp83) * -0.5 + _tmp122 * _tmp178 + _tmp125 * _tmp162 + _tmp165 * _tmp188
		D[2] =(trigamma(βᵅ) + _tmp178) * _tmp122 + -(((((-((((-(_tmp46 * _tmp54) * _tmp104 + _tmp137 * _tmp181) * _tmp112 + -(_tmp54 * _tmp9) + _tmp154 * _tmp187) * _tmp114 + _tmp115 * _tmp137 + _tmp154 * _tmp3 + _tmp185 * _tmp53 + _tmp43 * _tmp54 + _tmp48 * _tmp54)) + -(-(_tmp130 * _tmp138) * _tmp117) + _tmp127 * _tmp53) * _tmp118 + _tmp42) * _tmp77 + _tmp186 * _tmp73) * _tmp76 + _tmp154 * _tmp25) * 0.5) + -((-(((_tmp149 * _tmp27 + _tmp154 * _tmp75) * _tmp81 + _tmp134 * _tmp138)) * _tmp177 + _tmp135 * _tmp169) * _tmp50) + -(_tmp126 * _tmp53) + -(_tmp136 * _tmp166) + -(_tmp161 * _tmp53) + -(_tmp169 * _tmp81) * -0.5 + _tmp121 * _tmp178 + _tmp136 * _tmp188 + _tmp162 * _tmp53
		D[3] =(trigamma(αᵝ) + _tmp179) * _tmp139 + -(((((-(_tmp91 * _tmp92 * (_tmp93 + _tmp95)) + _tmp144 * _tmp93 + _tmp145 * _tmp95) * _tmp99 + -_tmp94 * _tmp31) * 0.5 * _tmp100 + _tmp20) * _tmp87 + _tmp100 * _tmp168 * _tmp31 * _tmp94)) + -(_tmp146 * _tmp94) + _tmp141 * _tmp179 + _tmp142 * _tmp93 + _tmp143 * _tmp95
		D[4] = (trigamma(βᵝ) + _tmp179) * _tmp141 + -(((((-((_tmp58 + _tmp59) * _tmp91 * _tmp92) + _tmp144 * _tmp58 + _tmp145 * _tmp59) * _tmp168 + -_tmp66 * 0.5 * _tmp31) * _tmp100 + _tmp20) * _tmp64 + _tmp100 * _tmp151 * _tmp184) * αᵝ) + -(_tmp146 * _tmp66) + _tmp139 * _tmp179 + _tmp142 * _tmp58 + _tmp143 * _tmp59
		nothing
    end


