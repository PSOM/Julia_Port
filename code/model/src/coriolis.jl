@kernel function k_coriolis!(si, sj, sk, ffc, bbc, u, v, w, drpx, drpy, uvis, vvis, wvis,
                             n, fac, fac2, ainv, betainv, fnhhy, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    si[i, j, k] = -ffc[i, j] * v[i, j, k, n] + fac * bbc[i, j] * w[i, j, k, n] + drpx[i, j, k] - EPS * uvis[i, j, k]
    sj[i, j, k] = ffc[i, j] * u[i, j, k, n] + drpy[i, j, k] - EPS * vvis[i, j, k]
    sk[i, j, k] = fnhhy * (-bbc[i, j] * u[i, j, k, n] - fac2 * (u[i, j, k, n] * u[i, j, k, n] + v[i, j, k, n] * v[i, j, k, n]) * ainv) - betainv * wvis[i, j, k]
  end
end

function coriolis(n, S=DEV)
  local fac = EPS * delta
  local fac2 = EPS * lambda
  local ainv = 1e0 / apr
  local betainv = 1e0 / beta
  launch!(S, k_coriolis!, (1:NI, 1:NJ, 1:NK), S.si, S.sj, S.sk, S.ffc, S.bbc, S.u, S.v, S.w,
          S.drpx, S.drpy, S.uvis, S.vvis, S.wvis, n, fac, fac2, ainv, betainv, fnhhy)
end
