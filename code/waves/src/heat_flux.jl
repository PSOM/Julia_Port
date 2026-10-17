
# Tdif(i,j,k) for one point: divergence of the short-wave flux, absorbed with the double-exponential
# profile of the water type (J_A, J_lambda1, J_lambda2) at the centre column; surface flux at NK, none
# through the bottom. The profile is recomputed per point (the Fortran keeps swrd, Kdfluxdzz per column).
@inline function hf_Kdfluxdzz(swr, zf, j, k, J_A, J_lambda1, J_lambda2)
    local swrd = swr[j] * (J_A * exp(zf[div(NI, 2), div(NJ, 2), k] * DL / J_lambda1) + (1 - J_A) * exp(zf[div(NI, 2), div(NJ, 2), k] * DL / J_lambda2))
    return DL * swrd / (R0 * 4187e0)
end

@kernel function k_heat_flux!(Tdif, swr, qloss, zf, Jac, wz, fac, J_A, J_lambda1, J_lambda2, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        if (k == NK)
            local Kdfluxdzt = DL * (swr[j] - qloss[j]) / (R0 * 4187e0)
            Tdif[i, j, NK] = fac * Jac[i, j, NK] * wz[i, j, NK] * Kdfluxdzt
        elseif (k == 1)
            local Kdfluxdzb = 0e0
            Tdif[i, j, 1] = fac * Jac[i, j, 1] * wz[i, j, 1] * Kdfluxdzb
        else
            Tdif[i, j, k] = fac * Jac[i, j, k] * wz[i, j, k] * (hf_Kdfluxdzz(swr, zf, j, k, J_A, J_lambda1, J_lambda2) - hf_Kdfluxdzz(swr, zf, j, k - 1, J_A, J_lambda1, J_lambda2))
        end
    end
end

"""
function heat_flux(Tdif, step, S=DEV)
    # arguments
    - `Tdif` 
"""
function heat_flux(Tdif, step, S=DEV)

    # This part deals with the water type coefficients
    local J_lambda1 = 0.6e0
    @fortSet("j_lambda1", J_lambda1, rc_kind)
    local J_lambda2 = 20e0
    @fortSet("j_lambda2", J_lambda2, rc_kind)
    local J_A = 0.62e0
    @fortSet("j_a", J_A, rc_kind)

    # This part deals with short wave radiation
    dfill!(S.swr, 0e0)
    dfill!(S.qloss, 0e0)


    local fac = 1e0 / (UL * DL * delta)
    launch!(S, k_heat_flux!, (1:NI, 1:NJ, 1:NK), Tdif, S.swr, S.qloss, S.zf, S.Jac, S.wz, fac, J_A, J_lambda1, J_lambda2)
end
