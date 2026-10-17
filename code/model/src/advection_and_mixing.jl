using .fortVar
using .cppdefs

# The kernels k_am_* are in advect.jl (shared with the experiment copies of this file, e.g. waves).

function advection_and_mixing(m, n, dtimel, step, S=DEV)

    local iv_compute_kz = 1

    local var = work(S, :am_var, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)

    local u_w = work(S, :am_u_w, rc_kind, NI, NJ, NK)
    local v_w = work(S, :am_v_w, rc_kind, NI, NJ, NK)
    local T_h = work(S, :am_T_h, rc_kind, NI, NJ, NK)

    heat_flux(T_h, step, S)

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

            @static if (cppdefs.implicit)
                # NOT ported (cppdefs.implicit is off): the implicit branch below runs on the host arrays
                local uvarx_advec = copy(uvarx)
                if (selvar < 5.5)
                    @views @. uvarx_advec[1:NI, 1:NJ, 1:NK] = uvarx[1:NI, 1:NJ, 1:NK] - vdh[1:NI, 1:NJ, 1:NK]
                end
            end

            if (selvar < 4.5)
                mixing_vertical(var, vdv, m, step, iv_compute_kz, selvar, S)
                iv_compute_kz = 0
            end

            # forcing: heat flux for T, wind stress for u and v
            local frc = selvar == 1 ? T_h : (selvar == 3 ? u_w : v_w)
            local usef = !cppdefs.implicit && (selvar == 1 || selvar == 3 || selvar == 4)
            launch!(S, k_am_tend!, rng, uvarx, vdh, vdv, frc, var, S.Jac, S.wz,
                    selvar < 5.5, selvar < 4.5, usef, !cppdefs.implicit && (selvar == 3 || selvar == 4), cbot)

            @static if (cppdefs.implicit)
                # NOT ported (cppdefs.implicit is off; mat_A..mat_E only exist with it)
                if (selvar == 1)
                    mat_D[:,:,:,selvar] = T_h[1:NI, 1:NJ, 1:NK]
                elseif (selvar == 3)
                    mat_D[:,:,:,selvar] = u_w[1:NI, 1:NJ, 1:NK]
                elseif (selvar == 4)
                    mat_D[:,:,:,selvar] = v_w[1:NI, 1:NJ, 1:NK]
                end
                local var0 = OffsetArray(zeros(rc_kind, (NI + 2, NJ + 2, NK + 2)), 0:NI+1, 0:NJ+1, 0:NK+1)
                if (selvar <= 5)
                    @views @. var0[0:NI+1, 0:NJ+1, 0:NK+1] = fields[selvar][0:NI+1, 0:NJ+1, 0:NK+1, 0]
                end
                if (selvar < 4.5)
                    for i in 1:NI
                        for j in 1:NJ
                            mat_A[i,j,1:NK, selvar] .*= dtimel * JacInv[i, j, 1:NK]
                            mat_B[i,j,1:NK, selvar] = (dtimel * JacInv[i, j, 1:NK] * mat_B[i,j,1:NK, selvar]) .- 1
                            mat_C[i,j,1:NK-1, selvar] .*= dtimel * JacInv[i, j, 1:NK-1]
                            mat_D[i,j,1:NK, selvar] .= 0e0 .- var0[i, j, 1:NK] - (dtimel * JacInv[i, j, 1:NK] * mat_D[i,j,1:NK, selvar]) + (dtimel * JacInv[i, j, 1:NK] * uvarx_advec[i,j,1:NK])
                            solve_tridiag(view(mat_A,i,j,1:NK, selvar), view(mat_B,i,j,1:NK, selvar), view(mat_C,i,j,1:NK, selvar), view(mat_D,i,j,1:NK, selvar), view(mat_test,i,j,1:NK, selvar), NK)
                        end
                    end
                    if (selvar == 1)
                        T[1:NI, 1:NJ, 1:NK, n] = mat_test[:,:,:,selvar]
                    end
                    if (selvar == 2)
                        s[1:NI, 1:NJ, 1:NK, n] = mat_test[:,:,:,selvar]
                    end
                    if (selvar == 3)
                        cx[1:NI, 1:NJ, 1:NK] = mat_test[:,:,:,selvar]
                    end
                    if (selvar == 4)
                        cy[1:NI, 1:NJ, 1:NK] = mat_test[:,:,:,selvar]
                    end
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
            end
            if (selvar == 5)
                launch!(S, k_am_upd3!, rng, S.cz, S.w, dtimel, S.JacInv, uvarx)
            end
            if (selvar > 5)
                launch!(S, k_am_upd_tr!, rng, S.Tr, selvar - 5, n, dtimel, S.JacInv, uvarx)
            end
        end
    end

end
