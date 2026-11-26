# hbc: coefficients and rhs of the free-surface equation at the boundary rows j=1 and j=NJ.
# Vertical sums are sequential (k=1..NK), as in hbc.f90.

@inline function hbc_colsum(a, i, j)
  local s = 0e0
  for k in 1:NK
    s += a[i, j, k]
  end
  return s
end
@inline function hbc_colsum(a, i, j, l)
  local s = 0e0
  for k in 1:NK
    s += a[i, j, k, l]
  end
  return s
end
@inline function hbc_colsum2(a, i)          # a(i,k), e.g. vfbcs
  local s = 0e0
  for k in 1:NK
    s += a[i, k]
  end
  return s
end

@kernel function k_hbc!(chf, fn, wfbcb, J2d, oldh, sifc, cxf, uf, hxn, gi, sjfc, cyf, vf, hyn, gj, vfbcs, vfbcn,
                        eg, kaphinv, dtinv, gprinv, constv, lo)
  i, r = @index(Global, NTuple)
  @inbounds begin
    local j = r == 1 ? 1 : NJ
    fn[i, j] = eg * (kaphinv * wfbcb[i, j] - J2d[i, j] * oldh[i, j] * dtinv)
    chf[1, i, j] = -eg * J2d[i, j] * dtinv
    for l in 2:9
      chf[l, i, j] = 0e0
    end
    let sumsif = hbc_colsum(sifc, i, j), sumcxf = hbc_colsum(cxf, i, j), sumuf = hbc_colsum(uf, i, j),
        sumhxn = hbc_colsum(hxn, i, j), sumgi1 = hbc_colsum(gi, i, j, 1), sumgi2 = hbc_colsum(gi, i, j, 2)
      fn[i, j] = fn[i, j] + eg * (sumcxf + constv * sumuf) - gprinv * sumsif - constv * sumhxn
      chf[1, i, j] -= sumgi1
      chf[2, i, j] += sumgi1
      chf[4, i, j] += 0.25e0 * sumgi2
      chf[5, i, j] -= 0.25e0 * sumgi2
      chf[6, i, j] += 0.25e0 * sumgi2
      chf[8, i, j] -= 0.25e0 * sumgi2
    end

    local im1 = (i == 1 ? NI : i - 1)
    let sumsif = hbc_colsum(sifc, im1, j), sumcxf = hbc_colsum(cxf, im1, j), sumuf = hbc_colsum(uf, im1, j),
        sumhxn = hbc_colsum(hxn, im1, j), sumgi1 = hbc_colsum(gi, im1, j, 1), sumgi2 = hbc_colsum(gi, im1, j, 2)
      fn[i, j] = fn[i, j] - eg * (sumcxf + constv * sumuf) + gprinv * sumsif + constv * sumhxn
      chf[1, i, j] -= sumgi1
      chf[3, i, j] += sumgi1
      chf[4, i, j] -= 0.25e0 * sumgi2
      chf[5, i, j] += 0.25e0 * sumgi2
      chf[7, i, j] -= 0.25e0 * sumgi2
      chf[9, i, j] += 0.25e0 * sumgi2
    end

    if (j == NJ)
      local sumvf = hbc_colsum2(vfbcn, i)
      fn[i, j] = fn[i, j] + eg * sumvf * kaphinv
    else
      local sumsjf = hbc_colsum(sjfc, i, j)
      local sumcyf = hbc_colsum(cyf, i, j)
      local sumvf = hbc_colsum(vf, i, j)
      local sumhyn = hbc_colsum(hyn, i, j)
      local sumgj1 = hbc_colsum(gj, i, j, 1)
      local sumgj2 = hbc_colsum(gj, i, j, 2)
      fn[i, j] = fn[i, j] + eg * (sumcyf + constv * sumvf) - gprinv * sumsjf - constv * sumhyn
      chf[1, i, j] -=          sumgj2
      chf[2, i, j] += 0.25e0 * sumgj1
      chf[3, i, j] -= 0.25e0 * sumgj1
      chf[4, i, j] +=          sumgj2
      chf[6, i, j] += 0.25e0 * sumgj1
      chf[7, i, j] -= 0.25e0 * sumgj1
    end

    if (j == 1)
      local sumvf = hbc_colsum2(vfbcs, i)
      fn[i, j] = fn[i, j] - eg * sumvf * kaphinv
    else
      local sumsjf = hbc_colsum(sjfc, i, j-1)
      local sumcyf = hbc_colsum(cyf, i, j-1)
      local sumvf = hbc_colsum(vf, i, j-1)
      local sumhyn = hbc_colsum(hyn, i, j-1)
      local sumgj1 = hbc_colsum(gj, i, j-1, 1)
      local sumgj2 = hbc_colsum(gj, i, j-1, 2)
      fn[i, j] = fn[i, j] - eg * (sumcyf + constv * sumvf) + gprinv * sumsjf + constv * sumhyn
      chf[1, i, j] -=          sumgj2
      chf[2, i, j] -= 0.25e0 * sumgj1
      chf[3, i, j] += 0.25e0 * sumgj1
      chf[5, i, j] +=          sumgj2
      chf[8, i, j] -= 0.25e0 * sumgj1
      chf[9, i, j] += 0.25e0 * sumgj1
    end
  end
end

function hbc(chf, fn, dtimel, S=DEV)
  local edt = EPS / dtimel
  local eg = edt / (gpr * kappah)
  local gprinv = 1e0 / (gpr * kappah)
  local constv = kaphinv - 1e0
  local dtinv = HDL / (dtimel * kappah)
  # work items (i, r): r = 1 -> row j=1, r = 2 -> row j=NJ
  launch!(S, k_hbc!, (1:NI, 1:2), chf, fn, S.wfbcb, S.J2d, S.oldh, S.sifc, S.cxf, S.uf, S.hxn, S.gi,
          S.sjfc, S.cyf, S.vf, S.hyn, S.gj, S.vfbcs, S.vfbcn, eg, kaphinv, dtinv, gprinv, constv)
end
