# Surface wind stress (stress_top_x = 0, stress_top_y a tanh profile in y, the same for every i) and
# the source terms udif, vdif it gives at the top level NK.

# one work item per (i, j): stress_top_y(1, j) of the Fortran is recomputed for every i (same value)
@kernel function k_wind_stress!(udif, vdif, stress_top, stress_top_x, stress_top_y, rho, Jac, wz,
                                yc, stressmax, yw, y0, yset, fac, fact, lo)
    i, j = @index(Global, NTuple) .+ lo
    @inbounds begin
        local sty
        if (j <= div(NJ, 2))
            sty = stressmax * 0.5 * (1e0 + tanh(((yc[j] * 1e3 - yset) / yw) * PI))
        else
            local yset2 = yc[NJ] * 1e3 - y0 - yw / 2e0
            sty = stressmax * 0.5 * (1e0 + tanh((-(yc[j] * 1e3 - yset2) / yw) * PI))
        end

        local stx = 0e0
        stress_top_x[i, j] = stx
        stress_top_y[i, j] = sty
        stress_top[i, j] = sqrt(stx * stx + sty * sty)
        local rhoinv = 1e0 / rho[i, j, NK]

        local Kdudzt = stx * rhoinv * fact
        local Kdvdzt = sty * rhoinv * fact

        udif[i, j, NK] = fac * Jac[i, j, NK] * wz[i, j, NK] * Kdudzt
        vdif[i, j, NK] = fac * Jac[i, j, NK] * wz[i, j, NK] * Kdvdzt
    end
end

function wind_stress(udif, vdif, step, S=DEV)
    dfill!(udif, 0e0)
    dfill!(vdif, 0e0)

    local stressmax = 0e0

    local iyw = 120
    local iy0 = 40

    local yw = rc_kind(iyw) * dy
    local y0 = rc_kind(iy0) * dy - dy / 2e0
    local yset = y0 + yw / 2e0

    local fac = 1e0 / (UL * DL * delta)
    local fact = DL / UL
    launch!(S, k_wind_stress!, (1:NI, 1:NJ), udif, vdif, S.stress_top, S.stress_top_x, S.stress_top_y, S.rho, S.Jac, S.wz,
            S.yc, stressmax, yw, y0, yset, fac, fact)
end
