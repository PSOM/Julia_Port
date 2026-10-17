using .fortVar
using .cppdefs

# Kz from the surface stress (Ekman-layer profile), all levels k = 0:NK
@kernel function k_mixv_kz!(Kz, stress_top, ffc, zf, KzmaxTr, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local ustar = sqrt(stress_top[i, j] / R0)
        local ff = ffc[i, j] * FPAR
        local Ekdepth = 0.4e0 * ustar / ff
        local zextent = 0.5e0 * Ekdepth
        local ztransit = -Ekdepth
        local zdep = zf[i, j, k] * DL
        local thy = (1e0 + tanh(((zdep - ztransit) / zextent) * PI)) * 0.5e0
        Kz[i, j, k] = max(0.01e0, thy) * KzmaxTr
    end
end

# vertical gradient on the face k+1/2 (dvardzfc), 0 at k = 0 and k = NK
@inline mixv_dzfc(wz, var, i, j, k) = (k == 0 || k == NK) ? 0e0 : wz[i, j, k] * (var[i, j, k+1] - var[i, j, k])

# Port of the column loop of waves/src/mixing_vertical.f90 (also fills the implicit-mixing
# matrices mat_A/B/C for this selvar); one work item per point
@kernel function k_mixing_vertical!(vardif, var, Jac, wz, Kz, mat_A, mat_B, mat_C, selvar, fac, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local dfactor = fac * Jac[i, j, k] * wz[i, j, k]
        vardif[i, j, k] = dfactor * (mixv_dzfc(wz, var, i, j, k) * Kz[i, j, k] - mixv_dzfc(wz, var, i, j, k - 1) * Kz[i, j, k-1])
        if (k == 1)
            mat_B[i, j, k, selvar] = (-1e0) * dfactor * (Kz[i, j, k] * wz[i, j, k])
            mat_C[i, j, k, selvar] = dfactor * Kz[i, j, k] * wz[i, j, k]
        elseif (k < NK)
            mat_A[i, j, k, selvar] = dfactor * Kz[i, j, k-1] * wz[i, j, k-1]
            mat_B[i, j, k, selvar] = (-1e0) * dfactor * (Kz[i, j, k] * wz[i, j, k] + Kz[i, j, k-1] * wz[i, j, k-1])
            mat_C[i, j, k, selvar] = dfactor * Kz[i, j, k] * wz[i, j, k]
        else
            mat_A[i, j, k, selvar] = dfactor * Kz[i, j, k-1] * wz[i, j, k-1]
            mat_B[i, j, k, selvar] = (-1e0) * dfactor * Kz[i, j, k-1] * wz[i, j, k-1]
        end
    end
end

function mixing_vertical(var, vardif, m, step, iv_compute_kzl, selvar, S=DEV)
    local Kzmax = 1e-3
    local KzmaxTr = 1e-3
    local facb = RR * DL
    local fac = 1e0 / (UL * DL * delta)
    local fact = DL / UL
    @static if (cppdefs.gotm_call)
        for j in 1:NJ
            for i in 1:NI
                @ccall "./PSOM_LIB.so".shearn2_(Ref(i)::Ref{Int}, Ref(j)::Ref{int})::Cvoid

                @ccall "./PSOM_LIB.so".couple_gotm_(Ref(i)::Ref{Int}, Ref(j)::Ref{int})::Cvoid
                if (selvar == 1 || selvar == 2)
                    Kz[i, j, 1:NK] = KzTr[i, j, 1:NK]
                elseif (selvar == 3 || selvar == 4 || selvar == 5)
                    Kz[i, j, 1:NK] = KzMom[i, j, 1:NK]
                end
            end
        end
    else
        if (iv_compute_kzl == 1)
            launch!(S, k_mixv_kz!, (1:NI, 1:NJ, 0:NK), S.Kz, S.stress_top, S.ffc, S.zf, KzmaxTr)
        end
    end

    launch!(S, k_mixing_vertical!, (1:NI, 1:NJ, 1:NK), vardif, var, S.Jac, S.wz, S.Kz,
            S.mat_A, S.mat_B, S.mat_C, selvar, fac)

end

function viscosity(dudz, dvdz, drdz, i, j)
    local n1 = 3
    local grho = 9.81 / R0
    local DLinv = DLinv
    local fac = (UL^2) / (DL^2)
    local RiCr = 0.7e0
    Kz[i, j, 0] = 0e0
    Kz[i, j, NK] = 0e0

    for k in 1:NK-1
        local bvfreq = -grho * drdz[k] * DLinv
        local vshear = ((dudz[k]^2) + (dvdz[k]^2)) * fac
        local Ri = vshear == 0 ? 100e0 : bvfreq / vshear
        if (Ri == 0)
            Kz[i, j, k] = 1e0
        elseif (Ri == RiCr)
            Kz[i, j, k] = (1e0 - (Ri^2) / (RiCr^2))^n1
        else
            Kz[i, j, k] = 1e0
        end
    end

end