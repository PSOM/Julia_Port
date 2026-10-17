# Source terms at the cell faces (sifc, sjfc): 3rd-order interpolation of the Coriolis and viscous terms,
# plus the pressure gradient (grpifc, grpjfc, p).

# i-faces i = 1:NI (periodic in i)
@kernel function k_srcface_sifc!(sifc, u, v, w, uvis, vvis, p, ux, uy, ffi, bbi, gqi, gqi3, Jifc, grpifc,
                                 n, fac, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local xa::rc_kind = 9e0 / 16e0
    local xb::rc_kind = -1e0 / 16e0

    local i0::Int = i
    local im1::Int = i-1
    local ip1::Int = i+1
    local ip2::Int = i+2

    if (i == 1)
      i0 = i
      im1 = NI
      ip1 = i+1
      ip2 = i+2
    elseif (i == NI - 1)
      i0 = i
      im1 = i-1
      ip1 = i+1
      ip2 = 1
    elseif (i == NI)
      i0 = i
      im1 = i-1
      ip1 = 1
      ip2 = 2
    end

    # metric terms at the face (periodic: face NI uses column 1)
    local iu = (i == NI) ? 1 : i + 1
    local uxi = 0.5e0 * (ux[iu, j] + ux[i, j])
    local uyi = 0.5e0 * (uy[iu, j] + uy[i, j])

    local uint = xa * (u[i0, j, k, n] + u[ip1, j, k, n]) + xb * (u[im1, j, k, n] + u[ip2, j, k, n])
    local vint = xa * (v[i0, j, k, n] + v[ip1, j, k, n]) + xb * (v[im1, j, k, n] + v[ip2, j, k, n])
    local wint = xa * (w[i0, j, k, n] + w[ip1, j, k, n]) + xb * (w[im1, j, k, n] + w[ip2, j, k, n])

    local vcif = EPS * (xa * (uvis[i0, j, k] + uvis[ip1, j, k]) + xb * (uvis[im1, j, k] + uvis[ip2, j, k]))
    local vcjf = EPS * (xa * (vvis[i0, j, k] + vvis[ip1, j, k]) + xb * (vvis[im1, j, k] + vvis[ip2, j, k]))

    local px = ((p[ip1, j, k] - p[i, j, k]) * gqi[i, j, k, 1]
    + 0.25e0 * (p[ip1, j+1, k] + p[i, j+1, k] - p[ip1, j-1, k] - p[i, j-1, k]) * gqi[i, j, k, 2]
    + 0.25e0 * (p[ip1, j, k+1] + p[i, j, k+1] - p[ip1, j, k-1] - p[i, j, k-1]) * gqi3[i, j, k])
    sifc[i, j, k] = ((uxi * (-ffi[i, j] * vint + fac * bbi[i, j] * wint - vcif)
    + uyi * ( ffi[i, j] * uint - vcjf)) * Jifc[i, j, k] + grpifc[i, j, k]) + px
  end
end

@kernel function k_srcface_sifc_bc!(sifc, lo)
  j, k = @index(Global, NTuple) .+ lo
  @inbounds sifc[0, j, k] = sifc[NI, j, k]
end

# j-faces j = 0:NJ (interior 1:NJ-1 with 3rd-order interpolation; at the solid boundaries j = 0, NJ
# extrapolation from the first two interior points)
@kernel function k_srcface_sjfc!(sjfc, u, v, w, uvis, vvis, p, vx, vy, ffj, bbj, ffc, gqj, gqj3, Jjfc, grpjfc,
                                 n, fac, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local vxj = 0.5e0 * (vx[i, j+1] + vx[i, j])
    local vyj = 0.5e0 * (vy[i, j+1] + vy[i, j])

    local py = ((p[i, j+1, k] - p[i, j, k]) * gqj[i, j, k, 2]
    + 0.25e0 * (p[i+1, j+1, k] + p[i+1, j, k] - p[i-1, j+1, k] - p[i-1, j, k]) * gqj[i, j, k, 1]
    + 0.25e0 * (p[i, j+1, k+1] + p[i, j, k+1] - p[i, j+1, k-1] - p[i, j, k-1]) * gqj3[i, j, k])

    if (j == 0 || j == NJ)
      local in = 1
      local inm1 = 2
      if (j == NJ)
        in = NJ
        inm1 = NJ-1
      end

      local vint = 0.5e0 * (3e0 * v[i, in, k, n] - v[i, inm1, k, n])
      local uint = 0.5e0 * (3e0 * u[i, in, k, n] - u[i, inm1, k, n])
      local wint = 0.5e0 * (3e0 * w[i, in, k, n] - w[i, inm1, k, n])

      local vcif = EPS * uvis[i, in, k]
      local vcjf = EPS * vvis[i, in, k]

      sjfc[i, j, k] = ((vxj * (-ffj[i, j] * vint + fac * bbj[i, j] * wint - vcif)
      + vyj * ( ffc[i, j] * uint - vcjf)) * Jjfc[i, j, k] + grpjfc[i, j, k]) + py
    else
      local j0 = j
      local jm1 = j-1
      local jp1 = j+1
      local jp2 = j+2
      local xa = 9e0/16e0
      local xb = -1e0/16e0

      if (j == 1)
        j0 = j
        jm1 = NJ
        jp1 = j+1
        jp2 = j+2
        xa = 0.5e0
        xb = 0e0
      elseif (j == NJ-1)
        j0 = j
        jm1 = j-1
        jp1 = NJ
        jp2 = 1
        xa = 0.5e0
        xb = 0e0
      end

      local vint = xa * (v[i, j0, k, n] + v[i, jp1, k, n]) + xb * (v[i, jm1, k, n] + v[i, jp2, k, n])
      local uint = xa * (u[i, j0, k, n] + u[i, jp1, k, n]) + xb * (u[i, jm1, k, n] + u[i, jp2, k, n])
      local wint = xa * (w[i, j0, k, n] + w[i, jp1, k, n]) + xb * (w[i, jm1, k, n] + w[i, jp2, k, n])
      local vcif = EPS * (xa * (uvis[i, j0, k] + uvis[i, jp1, k]) + xb * (uvis[i, jm1, k] + uvis[i, jp2, k]))
      local vcjf = EPS * (xa * (vvis[i, j0, k] + vvis[i, jp1, k]) + xb * (vvis[i, jm1, k] + vvis[i, jp2, k]))

      sjfc[i, j, k] = ((vxj * (-ffj[i, j] * vint + fac * bbj[i, j+1] * wint - vcif)
      + vyj * ( ffc[i, j] * uint - vcjf)) * Jjfc[i, j, k] + grpjfc[i, j, k]) + py
    end
  end
end

function srcface(n, step, S=DEV)
  local fac = EPS * delta

  launch!(S, k_srcface_sifc!, (1:NI, 1:NJ, 1:NK), S.sifc, S.u, S.v, S.w, S.uvis, S.vvis, S.p, S.ux, S.uy,
          S.ffi, S.bbi, S.gqi, S.gqi3, S.Jifc, S.grpifc, n, fac)
  launch!(S, k_srcface_sifc_bc!, (1:NJ, 1:NK), S.sifc)

  launch!(S, k_srcface_sjfc!, (1:NI, 0:NJ, 1:NK), S.sjfc, S.u, S.v, S.w, S.uvis, S.vvis, S.p, S.vx, S.vy,
          S.ffj, S.bbj, S.ffc, S.gqj, S.gqj3, S.Jjfc, S.grpjfc, n, fac)
end
