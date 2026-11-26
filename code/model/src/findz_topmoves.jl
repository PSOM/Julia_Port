#     -------------------------------------------------------
#     finds the value of z (non-dim by DL) given the value of sigma.
#     ht and dep are non-dim by DL.
#     At every (i,j), the column is divided into NK equal-depth cells.

# one work item per (i,j) column
@kernel function k_findzall!(zc, zf, h, D, dnkm1_top, dnkm1, dnkm1inv, epm1inv, pfac, dztop, dzbot, hdl, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    #  In the surface layer
    local hpd = h[i, j] * hdl + dztop
    for k in NK:NK+1
      local sigma = rc_kind(k) - 0.5
      zc[i, j, k] = (sigma - dnkm1_top) * hpd - dztop
    end
    for k in NK-1:NK+1
      local sigma = rc_kind(k)
      zf[i, j, k] = (sigma - dnkm1_top) * hpd - dztop
    end

    #  Below the surface layer
    for k in 0:NK-1
      local sigma = rc_kind(k) - 0.5
      @static if (cppdefs.fixed_bottom_thickness)
        sigma = rc_kind(k - 1) - 0.5
      end
      local xfac = (dnkm1 - sigma) * dnkm1inv

      @static if (cppdefs.fixed_bottom_thickness)
        zc[i, j, k] = (exp(pfac * xfac) - 1e0) * epm1inv * (D[i, j] + dzbot + dztop) - dztop
      else
        zc[i, j, k] = (exp(pfac * xfac) - 1e0) * epm1inv * (D[i, j] + dztop) - dztop
      end
    end
    for k in -1:NK-2
      local sigma = rc_kind(k)
      @static if (cppdefs.fixed_bottom_thickness)
        sigma = rc_kind(k - 1)
      end
      local xfac = (dnkm1 - sigma) * dnkm1inv

      @static if (cppdefs.fixed_bottom_thickness)
        zf[i, j, k] = (exp(pfac * xfac) - 1e0) * epm1inv * (D[i, j] + dzbot + dztop) - dztop
      else
        zf[i, j, k] = (exp(pfac * xfac) - 1e0) * epm1inv * (D[i, j] + dztop) - dztop
      end
    end

    @static if (cppdefs.fixed_bottom_thickness)
      #     For the bottom boundary layer
      #     -------------------------------
      for k in 0:1
        local sigma = rc_kind(k) - 0.5
        zc[i, j, k] = ((sigma - 0e0) / rc_kind(1)) * dzbot + D[i, j]
      end
      for k in -1:1
        local sigma = rc_kind(k)
        zf[i, j, k] = ((sigma - 0e0) / rc_kind(1)) * dzbot + D[i, j]
      end
    end
  end
end

function findzall(S=DEV)
  # around NK...
  local dnkm1_top = rc_kind(NK - 1)
  local dnkm1 = dnkm1_top
  local dnkm1inv = 1e0 / dnkm1

  # stretching of the vertical grid
  #    pfac is the stretching in z. higher pfac gives more points near surf.
  global pfac = 2.0e0
  local epm1inv = 1e0 / (exp(pfac) - 1e0)

  @static if (cppdefs.fixed_bottom_thickness)
    dnkm1 = rc_kind(NK - 1 - 1)
    dnkm1inv = 1e0 / dnkm1
  end
  local dzb = @static cppdefs.fixed_bottom_thickness ? dzbot : 0e0

  launch!(S, k_findzall!, (0:NI+1, 0:NJ+1), S.zc, S.zf, S.h, S.D, dnkm1_top, dnkm1, dnkm1inv, epm1inv,
          pfac, dztop, dzb, HDL)
end
