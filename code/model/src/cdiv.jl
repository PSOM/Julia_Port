@kernel function k_cdiv!(dv, u, v, w, ux, uy, vx, vy, wx, wy, wz, Jac, n, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local Udxi = 0.5e0 * ((u[i+1, j, k, n] * ux[i+1, j] + v[i+1, j, k, n] * uy[i+1, j]) * Jac[i+1, j, k]
                        - (u[i-1, j, k, n] * ux[i-1, j] + v[i-1, j, k, n] * uy[i-1, j]) * Jac[i-1, j, k])
    local Vdelta = 0.5e0 * ((u[i, j+1, k, n] * vx[i, j+1] + v[i, j+1, k, n] * vy[i, j+1]) * Jac[i, j+1, k]
                          - (u[i, j-1, k, n] * vx[i, j-1] + v[i, j-1, k, n] * vy[i, j-1]) * Jac[i, j-1, k])
    local Wdsig = 0.5e0 * ((u[i, j, k+1, n] * wx[i, j, k+1] + v[i, j, k+1, n] * wy[i, j, k+1] + EPS * w[i, j, k+1, n] * wz[i, j, k+1]) * Jac[i, j, k+1]
                         - (u[i, j, k-1, n] * wx[i, j, k-1] + v[i, j, k-1, n] * wy[i, j, k-1] + EPS * w[i, j, k-1, n] * wz[i, j, k-1]) * Jac[i, j, k-1])
    dv[i, j, k] = abs(Udxi + Vdelta + Wdsig)
  end
end

"max |divergence| of the cell-centre velocities"
function cdiv(dtimel, n, S=DEV)
  local dv = work(S, :div3, rc_kind, NI, NJ, NK)
  launch!(S, k_cdiv!, (1:NI, 1:NJ, 1:NK), dv, S.u, S.v, S.w, S.ux, S.uy, S.vx, S.vy, S.wx, S.wy, S.wz, S.Jac, n)
  return maximum(dv)
end
