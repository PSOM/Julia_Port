using .fortVar

# QUICK advection with the ULTIMATE limiter (port of advect.f90). Fluxes on the faces in x, y, z,
# then uvarx = flux divergence. advect also fills the halos of var (periodic in x, zero-gradient
# in y and z), which the callers keep using.

function ultim(up::rc_kind, ct::rc_kind, dn::rc_kind, gdn::rc_kind, gup::rc_kind, courinv::rc_kind, fc::rc_kind)
    local del = dn - up
    local adel = abs(del)
    local acurv = abs(gdn - gup)
    if (acurv >= adel)
        return ct
    else
        local ref = up + (ct - up) * courinv
        if (del > 0)
            fc = max(ct, fc)
            ref = min(ref, dn)
            fc = min(fc, ref)
        else
            fc = min(ct, fc)
            ref = max(ref, dn)
            fc = max(fc, ref)
        end
    end
    return fc
end

# differences of var between neighbours (dvarx, dvary, dvarz of the Fortran)
# x: periodic, dvarx(0) = dvarx(NI) = var(1) - var(NI)
@inline adv_dvx(var, i, j, k) = (i == 0 || i == NI) ? var[1, j, k] - var[NI, j, k] : var[i+1, j, k] - var[i, j, k]
# y: dvary(0) = dvary(NJ) = 0
@inline adv_dvy(var, i, j, k) = (j == 0 || j == NJ) ? 0e0 : var[i, j+1, k] - var[i, j, k]
# z: dvarz(0) = dvarz(NK) = dvarz(NK+1) = 0
@inline adv_dvz(var, i, j, k) = (k == 0 || k >= NK) ? 0e0 : var[i, j, k+1] - var[i, j, k]

@kernel function k_adv_xhalo!(var, lo)
    j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        var[NI+1, j, k] = var[1, j, k]
        var[0, j, k] = var[NI, j, k]
    end
end

@kernel function k_adv_yhalo!(var, lo)
    i, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        var[i, 0, k] = var[i, 1, k]
        var[i, NJ+1, k] = var[i, NJ, k]
    end
end

@kernel function k_adv_zhalo!(var, lo)
    i, j = @index(Global, NTuple) .+ lo
    @inbounds begin
        var[i, j, 0] = var[i, j, 1]
        var[i, j, NK+1] = var[i, j, NK]
    end
end

# flux through the face i+1/2 (i = 1:NI; face NI is also face 0)
@kernel function k_adv_xflux!(flx, var, uf, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local Eighth::rc_kind = 0.125e0
        local Half::rc_kind = 0.5e0
        local courinv::rc_kind = 2e0
        if (uf[i, j, k] > 0)
            local left = Eighth * (adv_dvx(var, i, j, k) - adv_dvx(var, i - 1, j, k))
            local ctr = Half * (var[i+1, j, k] + var[i, j, k])
            local fc = ctr - left
            fc = ultim(var[i-1, j, k], var[i, j, k], var[i+1, j, k], adv_dvx(var, i, j, k), adv_dvx(var, i - 1, j, k), courinv, fc)
            flx[i, j, k] = uf[i, j, k] * fc
        else
            local i2 = i == NI ? 0 : i
            local right = Eighth * (adv_dvx(var, i2 + 1, j, k) - adv_dvx(var, i2, j, k))
            local ctr = Half * (var[i2+1, j, k] + var[i, j, k])
            local fc = ctr - right
            fc = ultim(var[i2+2, j, k], var[i2+1, j, k], var[i, j, k], adv_dvx(var, i, j, k), adv_dvx(var, i2 + 1, j, k), courinv, fc)
            flx[i, j, k] = uf[i, j, k] * fc
        end
    end
end

# flux through the face j+1/2 (j = 0:NJ; zero at the walls j = 0, NJ)
@kernel function k_adv_yflux!(flx, var, vf, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local Eighth = 0.125e0
        local Half = 0.5e0
        local courinv = 2e0
        if (j == 0 || j == NJ)
            flx[i, j, k] = 0e0
        elseif (vf[i, j, k] >= 0)
            local left = Eighth * (adv_dvy(var, i, j, k) - adv_dvy(var, i, j - 1, k))
            local ctr = Half * (var[i, j+1, k] + var[i, j, k])
            local fc = ctr - left
            fc = ultim(var[i, j-1, k], var[i, j, k], var[i, j+1, k], adv_dvy(var, i, j, k), adv_dvy(var, i, j - 1, k), courinv, fc)
            flx[i, j, k] = vf[i, j, k] * fc
        else
            local right = Eighth * (adv_dvy(var, i, j + 1, k) - adv_dvy(var, i, j, k))
            local ctr = Half * (var[i, j+1, k] + var[i, j, k])
            local fc = ctr - right
            fc = ultim(var[i, j+2, k], var[i, j+1, k], var[i, j, k], adv_dvy(var, i, j, k), adv_dvy(var, i, j + 1, k), courinv, fc)
            flx[i, j, k] = vf[i, j, k] * fc
        end
    end
end

# flux through the face k+1/2 (k = 0:NK; zero at the bottom k = 0)
@kernel function k_adv_zflux!(flx, var, wf, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local Eighth = 0.125e0
        local Half = 0.5e0
        local courinv = 2e0
        if (k == 0)
            flx[i, j, k] = 0e0
        elseif (wf[i, j, k] >= 0)
            local left = Eighth * (adv_dvz(var, i, j, k) - adv_dvz(var, i, j, k - 1))
            local ctr = Half * (var[i, j, k+1] + var[i, j, k])
            local fc = ctr - left
            fc = ultim(var[i, j, k-1], var[i, j, k], var[i, j, k+1], adv_dvz(var, i, j, k), adv_dvz(var, i, j, k - 1), courinv, fc)
            flx[i, j, k] = wf[i, j, k] * fc
        else
            if (k == NK)
                flx[i, j, k] = wf[i, j, k] * Half * (var[i, j, k+1] + var[i, j, k])
            else
                local right = Eighth * (adv_dvz(var, i, j, k + 1) - adv_dvz(var, i, j, k))
                local ctr = Half * (var[i, j, k+1] + var[i, j, k])
                local fc = ctr - right
                fc = ultim(var[i, j, k+2], var[i, j, k+1], var[i, j, k], adv_dvz(var, i, j, k), adv_dvz(var, i, j, k + 1), courinv, fc)
                flx[i, j, k] = wf[i, j, k] * fc
            end
        end
    end
end

# uvarx = x divergence, then += y divergence, then += z divergence (in this order, as in the Fortran)
@kernel function k_adv_div!(uvarx, fx, fy, fz, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local fxm = i == 1 ? fx[NI, j, k] : fx[i-1, j, k]     # varflx(0) = varflx(NI)
        local t = fx[i, j, k] - fxm
        t += (fy[i, j, k] - fy[i, j-1, k])
        t += (fz[i, j, k] - fz[i, j, k-1])
        uvarx[i, j, k] = t
    end
end

function advect(var, uvarx, S=DEV)
    local fx = work(S, :advect_fx, rc_kind, 0:NI, 0:NJ, 0:NK)
    local fy = work(S, :advect_fy, rc_kind, 0:NI, 0:NJ, 0:NK)
    local fz = work(S, :advect_fz, rc_kind, 0:NI, 0:NJ, 0:NK)

    # halos of var (each halo kernel reads only interior points)
    launch!(S, k_adv_xhalo!, (1:NJ, 1:NK), var)
    launch!(S, k_adv_yhalo!, (1:NI, 1:NK), var)
    launch!(S, k_adv_zhalo!, (1:NI, 1:NJ), var)

    launch!(S, k_adv_xflux!, (1:NI, 1:NJ, 1:NK), fx, var, S.uf)
    launch!(S, k_adv_yflux!, (1:NI, 0:NJ, 1:NK), fy, var, S.vf)
    launch!(S, k_adv_zflux!, (1:NI, 1:NJ, 0:NK), fz, var, S.wf)

    launch!(S, k_adv_div!, (1:NI, 1:NJ, 1:NK), uvarx, fx, fy, fz)
end
