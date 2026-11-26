# Cell-centre velocities cx, cy (hydrostatic update) and cz from them and czf.
# cz at a point uses the new cx, cy at the same point, so it is computed by the same work item.
@kernel function k_uvchy!(cx, cy, cz, h, p, ux, uy, vx, vy, wx, wy, wz, gradhn, si, sj, czf, Jac,
                          dte, kappah, kaph1, qpr, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local hxi = 0.5e0 * (h[i+1, j] - h[i-1, j])
    local heta = 0.5e0 * (h[i, j+1] - h[i, j-1])
    local hx = ux[i, j] * hxi + vx[i, j] * heta
    local hy = uy[i, j] * hxi + vy[i, j] * heta

    local pxi = 0.5e0 * (p[i+1, j, k] - p[i-1, j, k])
    local peta = 0.5e0 * (p[i, j+1, k] - p[i, j-1, k])
    if (j == 1)
      peta = p[i, j+1, k] - p[i, j, k]
    elseif (j == NJ)
      peta = p[i, j, k] - p[i, j-1, k]
    end
    local psig = 0.5e0 * (p[i, j, k+1] - p[i, j, k-1])
    local px = ux[i, j] * pxi + vx[i, j] * peta + wx[i, j, k] * psig
    local py = uy[i, j] * pxi + vy[i, j] * peta + wy[i, j, k] * psig

    cx[i, j, k] = cx[i, j, k] - dte * (gpr * (kappah * hx + kaph1 * gradhn[i, j, 1]) + si[i, j, k] + qpr * px)
    cy[i, j, k] = cy[i, j, k] - dte * (gpr * (kappah * hy + kaph1 * gradhn[i, j, 2]) + sj[i, j, k] + qpr * py)

    local wfk = 0.5e0 * (czf[i, j, k] + czf[i, j, k-1])
    cz[i, j, k] = (wfk / Jac[i, j, k] - cx[i, j, k] * wx[i, j, k] - cy[i, j, k] * wy[i, j, k]) / (EPS * wz[i, j, k])
  end
end

# gradhn is read by k_uvchy! at all levels, so it is updated in a separate launch
@kernel function k_uvchy_gradhn!(gradhn, h, ux, uy, vx, vy, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local hxi = 0.5e0 * (h[i+1, j] - h[i-1, j])
    local heta = 0.5e0 * (h[i, j+1] - h[i, j-1])
    gradhn[i, j, 1] = ux[i, j] * hxi + vx[i, j] * heta
    gradhn[i, j, 2] = uy[i, j] * hxi + vy[i, j] * heta
  end
end

function uvchy(dtimel, S=DEV)
  local dte = dtimel / EPS
  local kaph1 = 1e0 - kappah

  launch!(S, k_uvchy!, (1:NI, 1:NJ, 1:NK), S.cx, S.cy, S.cz, S.h, S.p, S.ux, S.uy, S.vx, S.vy, S.wx, S.wy, S.wz,
          S.gradhn, S.si, S.sj, S.czf, S.Jac, dte, kappah, kaph1, qpr)
  launch!(S, k_uvchy_gradhn!, (1:NI, 1:NJ), S.gradhn, S.h, S.ux, S.uy, S.vx, S.vy)
end
