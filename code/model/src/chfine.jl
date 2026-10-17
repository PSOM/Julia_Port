# chfine: coefficients chf (9-point stencil) and rhs fn of the free-surface equation. The boundary
# rows j=1 and j=NJ are done by hbc; this file does the vertical sums and the interior rows.

# vertical sums (sequential k=1..NK) of the face quantities, for j=1..NJ-1. Columns of sums:
#   1 sifc  2 sjfc  3 cxf  4 cyf  5 uf  6 vf  7 hxn  8 hyn  9 gi(.,1)  10 gi(.,2)  11 gj(.,1)  12 gj(.,2)
@kernel function k_chfine_sums!(sums, sifc, sjfc, cxf, cyf, uf, vf, hxn, hyn, gi, gj, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local s1 = 0e0; local s2 = 0e0; local s3 = 0e0; local s4 = 0e0; local s5 = 0e0; local s6 = 0e0
    local s7 = 0e0; local s8 = 0e0; local s9 = 0e0; local s10 = 0e0; local s11 = 0e0; local s12 = 0e0
    for k in 1:NK
      s1 += sifc[i, j, k]
      s2 += sjfc[i, j, k]
      s3 += cxf[i, j, k]
      s4 += cyf[i, j, k]
      s5 += uf[i, j, k]
      s6 += vf[i, j, k]
      s7 += hxn[i, j, k]
      s8 += hyn[i, j, k]
      s9 += gi[i, j, k, 1]
      s10 += gi[i, j, k, 2]
      s11 += gj[i, j, k, 1]
      s12 += gj[i, j, k, 2]
    end
    sums[i, j, 1] = s1; sums[i, j, 2] = s2; sums[i, j, 3] = s3; sums[i, j, 4] = s4
    sums[i, j, 5] = s5; sums[i, j, 6] = s6; sums[i, j, 7] = s7; sums[i, j, 8] = s8
    sums[i, j, 9] = s9; sums[i, j, 10] = s10; sums[i, j, 11] = s11; sums[i, j, 12] = s12
  end
end

# interior rows j=2..NJ-1 (periodic in x: i-1 of i=1 is NI)
@kernel function k_chfine!(chf, fn, sums, wfbcb, J2d, oldh, edtg, kaphinv, dtinv, constv, gpkinv, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local im1 = i == 1 ? NI : i - 1
    # sums[., ., c]: c = 1 sif, 2 sjf, 3 cxf, 4 cyf, 5 uf, 6 vf, 7 hxn, 8 hyn, 9/10 gi, 11/12 gj
    fn[i, j] = (edtg * (kaphinv * wfbcb[i, j] - J2d[i, j] * oldh[i, j] * dtinv
    + sums[i, j, 3] - sums[im1, j, 3] + sums[i, j, 4] - sums[i, j - 1, 4]
    + constv * (sums[i, j, 5] - sums[im1, j, 5] + sums[i, j, 6] - sums[i, j-1, 6]))
    - gpkinv * (sums[i, j, 1] - sums[im1, j, 1] + sums[i, j, 2] - sums[i, j-1, 2])
    - constv * (sums[i, j, 7] - sums[im1, j, 7] + sums[i, j, 8] - sums[i, j-1, 8]))

    chf[1, i, j] = -edtg * J2d[i, j] * dtinv - (sums[i, j, 9] + sums[im1, j, 9] + sums[i, j, 12] + sums[i, j-1, 12])
    chf[2, i, j] = sums[i, j, 9] + 0.25e0 * ( sums[i, j, 11] - sums[i, j-1, 11])
    chf[3, i, j] = sums[im1, j, 9] + 0.25e0 * (-sums[i, j, 11] + sums[i, j-1, 11])
    chf[4, i, j] = sums[i, j, 12] + 0.25e0 * (-sums[im1, j, 10] + sums[i, j, 10])
    chf[5, i, j] = sums[i, j-1, 12] + 0.25e0 * (-sums[i, j, 10] + sums[im1, j, 10])
    chf[6, i, j] = 0.25e0 * (sums[i, j, 10] + sums[i, j, 11])
    chf[7, i, j] = -0.25e0 * (sums[im1, j, 10] + sums[i, j, 11])
    chf[8, i, j] = -0.25e0 * (sums[i, j, 10] + sums[i, j-1, 11])
    chf[9, i, j] = 0.25e0 * (sums[im1, j, 10] + sums[i, j-1, 11])
  end
end

function chfine(dtimel, chf, fn, S=DEV)
  local sums = work(S, :chfine_sums, rc_kind, NI, NJ, 12)

  local edt = EPS / dtimel
  local edtg = edt / (gpr * kappah)
  local gpkinv = 1e0 / (gpr * kappah)
  local dtinv = HDL / (dtimel * kappah)
  local constv = kaphinv - 1e0

  hbc(chf, fn, dtimel, S)

  launch!(S, k_chfine_sums!, (1:NI, 1:NJ-1), sums, S.sifc, S.sjfc, S.cxf, S.cyf, S.uf, S.vf, S.hxn, S.hyn, S.gi, S.gj)
  launch!(S, k_chfine!, (1:NI, 2:NJ-1), chf, fn, sums, S.wfbcb, S.J2d, S.oldh, edtg, kaphinv, dtinv, constv, gpkinv)
end
