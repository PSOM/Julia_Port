using .fortVar

# Horizontal diffusion (port of mixing_horizontal.f90): Ky in y (no flux through the walls j = 1/2,
# NJ+1/2), Kx in x (periodic), times Jac / (UL*LEN).

# Ky-weighted gradient on the face j+1/2 (dvardyfc), 0 at j = 0, NJ
@inline mixh_dyfc(vy, var, i, j, k) = (j == 0 || j == NJ) ? 0e0 : 0.5 * (vy[i, j] + vy[i, j+1]) * (var[i, j+1, k] - var[i, j, k])
# gradient on the face i+1/2 (dvardxfc), periodic: face 0 = face NI
@inline mixh_dxfc(ux, var, i, j, k) = (i == 0 || i == NI) ? 0.5 * (ux[NI, j] + ux[1, j]) * (var[1, j, k] - var[NI, j, k]) :
                                      0.5 * (ux[i, j] + ux[i+1, j]) * (var[i+1, j, k] - var[i, j, k])

@kernel function k_mixing_horizontal!(vardif, var, ux, vy, Jac, Kx, Ky, fac, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        local t = Ky * vy[i, j] * (mixh_dyfc(vy, var, i, j, k) - mixh_dyfc(vy, var, i, j - 1, k))
        t += Kx * ux[i, j] * (mixh_dxfc(ux, var, i, j, k) - mixh_dxfc(ux, var, i - 1, j, k))
        vardif[i, j, k] = t * (fac * Jac[i, j, k])
    end
end

function mixing_horizontal(var, vardif, S=DEV)
    local fac = 1e0 / (UL * LEN)
    launch!(S, k_mixing_horizontal!, (1:NI, 1:NJ, 1:NK), vardif, var, S.ux, S.vy, S.Jac, Kx, Ky, fac)
end
