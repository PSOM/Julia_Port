# Port of waves/src/wind_stress.f90.
# udif(NI,NJ,NK), vdif(NI,NJ,NK): source terms from the surface wind stress (top level only).
# stress_top, stress_top_x, stress_top_y are the header arrays (NI,NJ).
# yw, ymid, yset are INTEGER in the Fortran.

# one work item per (i, j): stress_top_y(1, j) of the Fortran is recomputed for every i (same value)
@kernel function k_wind_stress!(udif, vdif, stress_top, stress_top_x, stress_top_y, rho, Jac, wz,
                                stressmax, yw, step, fac, fact, lo)
    i, j = @index(Global, NTuple) .+ lo
    @inbounds begin
        #*********************************
        # COMPUTATION OF THE WIND STRESS
        local sty
        local yyy = rc_kind(j) - 0.5
        if (j <= div(NJ, 2))
            local yset = 30
            sty = stressmax * 0.5 * (1e0 + tanh(((yyy - yset) / yw) * PI))
        else
            local yset = NJ - 30
            sty = stressmax * 0.5 * (1e0 + tanh((-(yyy - yset) / yw) * PI))
        end
        if (abs(sty - stressmax) < 0.001)
            sty = stressmax
        end
        if (abs(sty) < 0.001)
            sty = 0e0
        end
        if (step > 240)
            sty = 0e0
        end

        #************************************
        # COMPUTATION OF THE SOURCE TERM

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

    local stressmax = 0.26e0
    local yw = 50

    local fac = 1e0 / (UL * DL * delta)
    local fact = DL / UL
    launch!(S, k_wind_stress!, (1:NI, 1:NJ), udif, vdif, S.stress_top, S.stress_top_x, S.stress_top_y, S.rho, S.Jac, S.wz,
            stressmax, yw, step, fac, fact)
end
