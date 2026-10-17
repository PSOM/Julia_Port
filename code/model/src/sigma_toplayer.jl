# one work item per (i,j) column (only the top layer moves)
@kernel function k_sigma_col!(wz, Jac, JacInv, wx, wy, wt, wzk, gqk, g13, g23, h, ux, uy, vx, vy, J2d, hdt,
                              dnkm1, be2, hdl, dztop, qpr, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    # All these variables are functions of time
    local hpd = h[i, j] * hdl + dztop
    local hpdinv = 1e0 / hpd

    # Computation of hu:
    local hu = 0e0
    if (i == 0)
      hu = hdl * (h[i+1, j] - h[NI-1, j])
    elseif (i == NI + 1)
      hu = hdl * (h[2, j] - h[i-1, j])
    else
      hu = 0.5e0 * hdl * (h[i+1, j] - h[i-1, j])
    end

    # Computation of hv:
    local hv = 0e0
    if (j == 0)
      hv = hdl * (h[i, j+1] - h[i, j])
    elseif (j == NJ + 1)
      hv = hdl * (h[i, j] - h[i, j-1])
    else
      hv = 0.5e0 * hdl * (h[i, j+1] - h[i, j-1])
    end

    # Computation of hx and hy:
    local hx = hu * ux[i, j] + hv * vx[i, j]
    local hy = hu * uy[i, j] + hv * vy[i, j]

    #     wz is not a function of depth when the sigma lines are equally spa
    #     Hence wz is wz(i,j,time). For a stretched grid wz would be w(i,j,k
    #     Then Jac which is now Jac(i,j,time) would  become  Jac(i,j,k,time)

    # Computation of wx, wy, g13, g23:
    for k in NK:NK+1
      wz[i, j, k] = hpdinv
      Jac[i, j, k] = J2d[i, j] / wz[i, j, k]
      #     All these variables are functions of time and depth
      #     now hdt computed in vhydro already contains HDL
      local sig = rc_kind(k) - 0.5e0
      wx[i, j, k] = (dnkm1 - sig) * hx * hpdinv
      wy[i, j, k] = (dnkm1 - sig) * hy * hpdinv
      g13[i, j, k] = ux[i, j] * wx[i, j, k] + uy[i, j] * wy[i, j, k]
      g23[i, j, k] = vx[i, j] * wx[i, j, k] + vy[i, j] * wy[i, j, k]
    end

    for k in NK-1:NK
      local sig = rc_kind(k)
      wt[i, j, k] = (dnkm1 - sig) * hdt[i, j] * hpdinv
    end

    wzk[i, j, NK] = hpdinv
    wzk[i, j, NK-1] = 0.5e0 * (wz[i, j, NK] + wz[i, j, NK-1])

    for k in NK-1:NK
      local sig = rc_kind(k)
      local wxk = (dnkm1 - sig) * hx * hpdinv
      local wyk = (dnkm1 - sig) * hy * hpdinv
      gqk[i, j, k, 1] = qpr * Jac[i, j, k] * (ux[i, j] * wxk + uy[i, j] * wyk)
      gqk[i, j, k, 2] = qpr * Jac[i, j, k] * (vx[i, j] * wxk + vy[i, j] * wyk)
      gqk[i, j, k, 3] = Jac[i, j, k] * (qpr * (wxk * wxk + wyk * wyk) + be2 * wz[i, j, k] * wz[i, j, k])
    end

    if (i <= NI && j <= NJ)
      for k in NK:NK+1
        JacInv[i, j, k] = 1e0 / Jac[i, j, k]
      end
    end
  end
end

# i-faces of the top layer
@kernel function k_sigma_ifc!(Jifc, gi, gqi, gi3, gqi3, Jac, g11, g12, g13, qpr, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    Jifc[i, j, NK] = 0.5e0 * (Jac[i, j, NK] + Jac[i+1, j, NK])
    gi[i, j, NK, 1] = 0.5e0 * (g11[i, j] + g11[i+1, j]) * Jifc[i, j, NK]
    gi[i, j, NK, 2] = 0.5e0 * (g12[i, j] + g12[i+1, j]) * Jifc[i, j, NK]
    gqi[i, j, NK, 1] = qpr * gi[i, j, NK, 1]
    gqi[i, j, NK, 2] = qpr * gi[i, j, NK, 2]
    gi3[i, j, NK] = 0.5e0 * (g13[i, j, NK] + g13[i+1, j, NK]) * Jifc[i, j, NK]
    gqi3[i, j, NK] = qpr * gi3[i, j, NK]
  end
end

# j-faces of the top layer
@kernel function k_sigma_jfc!(Jjfc, gj, gqj, gj3, gqj3, Jac, g12, g22, g23, qpr, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    Jjfc[i, j, NK] = 0.5e0 * (Jac[i, j, NK] + Jac[i, j+1, NK])
    gj[i, j, NK, 1] = 0.5e0 * (g12[i, j] + g12[i, j+1]) * Jjfc[i, j, NK]
    gj[i, j, NK, 2] = 0.5e0 * (g22[i, j] + g22[i, j+1]) * Jjfc[i, j, NK]
    gqj[i, j, NK, 1] = qpr * gj[i, j, NK, 1]
    gqj[i, j, NK, 2] = qpr * gj[i, j, NK, 2]
    gj3[i, j, NK] = 0.5e0 * (g23[i, j, NK] + g23[i, j+1, NK]) * Jjfc[i, j, NK]
    gqj3[i, j, NK] = qpr * gj3[i, j, NK]
  end
end

function sigma(S=DEV)
  # g13, g23 are only used at k = NK (written for NK:NK+1 before they are read)
  local g13 = work(S, :sigma_g13, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)
  local g23 = work(S, :sigma_g23, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)

  local dnkm1 = rc_kind(NK-1)

  local be2 = beta * EPS * EPS

  launch!(S, k_sigma_col!, (0:NI+1, 0:NJ+1), S.wz, S.Jac, S.JacInv, S.wx, S.wy, S.wt, S.wzk, S.gqk, g13, g23,
          S.h, S.ux, S.uy, S.vx, S.vy, S.J2d, S.hdt, dnkm1, be2, HDL, dztop, qpr)
  launch!(S, k_sigma_ifc!, (0:NI, 1:NJ), S.Jifc, S.gi, S.gqi, S.gi3, S.gqi3, S.Jac, S.g11, S.g12, g13, qpr)
  launch!(S, k_sigma_jfc!, (1:NI, 0:NJ), S.Jjfc, S.gj, S.gqj, S.gj3, S.gqj3, S.Jac, S.g12, S.g22, g23, qpr)
end
