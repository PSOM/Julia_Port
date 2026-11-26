using .fortVar
using .cppdefs

# The kernels k_am_* are in model/src/advect.jl.

# mat_D(:, :, :, selvar) = src
@kernel function k_am_matD!(mat_D, src, selvar, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds mat_D[i, j, k, selvar] = src[i, j, k]
end

function advection_and_mixing(m, n, dtimel, step, S=DEV)

    local iv_compute_kz = 1

    local var = work(S, :am_var, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)

    dfill!(S.mat_A, 0e0)
    dfill!(S.mat_B, 0e0)
    dfill!(S.mat_C, 0e0)
    dfill!(S.mat_D, 0e0)
    dfill!(S.mat_test, 0e0)
    local u_w = work(S, :am_u_w, rc_kind, NI, NJ, NK)
    local v_w = work(S, :am_v_w, rc_kind, NI, NJ, NK)

    wind_stress(u_w, v_w, step, S)

    # uvarx: tendency (advection, then mixing and forcing); vdh, vdv: horizontal and vertical mixing
    # (vardif of the Fortran). Every point that is read is written first, so no zeroing is needed.
    local uvarx = work(S, :am_uvarx, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)
    local vdh = work(S, :am_vdh, rc_kind, NI, NJ, NK)
    local vdv = work(S, :am_vdv, rc_kind, NI, NJ, NK)
    local rng = (1:NI, 1:NJ, 1:NK)
    local cbot = RR * (1e0 / (UL * delta))
    local fields = (S.T, S.s, S.u, S.v, S.w)

    for selvar in 1:5+ntr
        @fortSet("selvar", selvar, rc_kind)

        # (av_comp of the Fortran: rhoonly skips T)
        if !(cppdefs.rhoonly && selvar == 1)
            if (selvar <= 5)
                launch!(S, k_am_load!, (0:NI+1, 0:NJ+1, 0:NK+1), var, fields[selvar], m)
            else
                launch!(S, k_am_load_tr!, (0:NI+1, 0:NJ+1, 0:NK+1), var, S.Tr, selvar - 5, m)
            end

            advect(var, uvarx, S)

            if (selvar < 5.5)
                mixing_horizontal(var, vdh, S)
            end

            @static if (cppdefs.gotm_im)
                # NOT ported (gotm_im is off): uvarx_advec = advection - horizontal mixing, host only
                local uvarx_advec = copy(uvarx)
                if (selvar < 5.5)
                    @views @. uvarx_advec[1:NI, 1:NJ, 1:NK] = uvarx[1:NI, 1:NJ, 1:NK] - vdh[1:NI, 1:NJ, 1:NK]
                end
            end

            if (selvar < 4.5)
                mixing_vertical(var, vdv, m, step, iv_compute_kz, selvar, S)
                iv_compute_kz = 0
            end

            # forcing (wind stress for u and v) and bottom drag (u and v, k = 1)
            local isuv = selvar == 3 || selvar == 4
            launch!(S, k_am_tend!, rng, uvarx, vdh, vdv, selvar == 3 ? u_w : v_w, var, S.Jac, S.wz,
                    selvar < 5.5, selvar < 4.5, isuv, isuv, cbot)
            if (selvar == 3)
                launch!(S, k_am_matD!, rng, S.mat_D, u_w, selvar)
            end
            if (selvar == 4)
                launch!(S, k_am_matD!, rng, S.mat_D, v_w, selvar)
            end

            @static if (cppdefs.gotm_im)   # implicit vertical mixing; NOT validated (off in the Fortran reference), NOT ported
                if (selvar < 6)
                    for i in 1:NI
                        for j in 1:NJ
                            @views mat_A[i, j, 1:NK, selvar] .= (-1e0) .* dtimel .* JacInv[i, j, 1:NK] .* mat_A[i, j, 1:NK, selvar]
                            @views mat_B[i, j, 1:NK, selvar] .= (1e0 .- (dtimel .* (JacInv[i, j, 1:NK] .* mat_B[i, j, 1:NK, selvar])))
                            @views mat_C[i, j, 1:NK-1, selvar] .= (-1e0) .* (dtimel .* (JacInv[i, j, 1:NK-1] .* mat_C[i, j, 1:NK-1, selvar]))
                            @views mat_D[i, j, 1:NK, selvar] .= dtimel .* (JacInv[i, j, 1:NK] .* (mat_D[i, j, 1:NK, selvar] .- uvarx_advec[i, j, 1:NK])) .+ var[i, j, 1:NK]
                            solve_tridiag(view(mat_A, i, j, 1:NK, selvar), view(mat_B, i, j, 1:NK, selvar), view(mat_C, i, j, 1:NK, selvar), view(mat_D, i, j, 1:NK, selvar), view(mat_test, i, j, 1:NK, selvar), NK)
                        end
                    end
                end
                if (selvar == 1)
                    T[1:NI, 1:NJ, 1:NK, n] = mat_test[:, :, :, selvar]
                end
                if (selvar == 2)
                    s[1:NI, 1:NJ, 1:NK, n] = mat_test[:, :, :, selvar]
                end
                if (selvar == 3)
                    cx[1:NI, 1:NJ, 1:NK] = mat_test[:, :, :, selvar]
                end
                if (selvar == 4)
                    cy[1:NI, 1:NJ, 1:NK] = mat_test[:, :, :, selvar]
                end
                if (selvar == 5)
                    cz[1:NI, 1:NJ, 1:NK] = mat_test[:, :, :, selvar]
                end

            else
                if (selvar == 1)
                    launch!(S, k_am_upd4!, rng, S.T, S.T, n, dtimel, S.JacInv, uvarx)
                end
                if (selvar == 2)
                    launch!(S, k_am_upd4!, rng, S.s, S.s, n, dtimel, S.JacInv, uvarx)
                end
                if (selvar == 3)
                    launch!(S, k_am_upd3!, rng, S.cx, S.u, dtimel, S.JacInv, uvarx)
                end
                if (selvar == 4)
                    launch!(S, k_am_upd3!, rng, S.cy, S.v, dtimel, S.JacInv, uvarx)
                end
                if (selvar == 5)
                    launch!(S, k_am_upd3!, rng, S.cz, S.w, dtimel, S.JacInv, uvarx)
                end
            end

            # (the cz update belongs to selvar==5 only, done above; the Julia port also overwrote cz here
            #  for every selvar, i.e. with the tracer tendency on the last pass)
            if (selvar > 5)
                launch!(S, k_am_upd_tr!, rng, S.Tr, selvar - 5, n, dtimel, S.JacInv, uvarx)
            end
        end
    end

end
