# vertical sums are sequential (k=1..NK), as in hbc.f90 (Base.sum may reorder the additions)
function hbc(chf,fn,dtimel) 
  local kaph1 = 1e0 - kappah
  local edt = EPS / dtimel
  local eg = edt / (gpr * kappah)
  local gprinv = 1e0 / (gpr * kappah)
  local constv = kaphinv - 1e0
  local dtinv = HDL / (dtimel * kappah)

  for i in 1:NI
    for j in 1:NJ
      if (j == 1 || j == NJ)
        fn[i, j] = eg * (kaphinv * wfbcb[i, j] - J2d[i, j] * oldh[i, j] * dtinv)
        chf[1, i, j] = -eg * J2d[i, j] * dtinv
        chf[2:9, i, j] .= 0e0
        local sumsif = seqsum(@view sifc[i, j, 1:NK])
        local sumcxf = seqsum(@view cxf[i, j, 1:NK])
        local sumuf = seqsum(@view uf[i, j, 1:NK])
        local sumhxn = seqsum(@view hxn[i, j, 1:NK])
        local sumgi = (seqsum(@view gi[i, j, 1:NK, 1]), seqsum(@view gi[i, j, 1:NK, 2]))
        
        fn[i, j] = fn[i, j] + eg * (sumcxf + constv * sumuf) - gprinv * sumsif - constv * sumhxn
        chf[1, i, j] -= sumgi[1]
        chf[2, i, j] += sumgi[1]
        chf[4, i, j] += 0.25e0 * sumgi[2]
        chf[5, i, j] -= 0.25e0 * sumgi[2]
        chf[6, i, j] += 0.25e0 * sumgi[2]
        chf[8, i, j] -= 0.25e0 * sumgi[2]
        
        local im1 = (i == 1 ? NI : i - 1)
        
        local sumsif = seqsum(@view sifc[im1, j, 1:NK])
        local sumcxf = seqsum(@view cxf[im1, j, 1:NK])
        local sumuf = seqsum(@view uf[im1, j, 1:NK])
        local sumhxn = seqsum(@view hxn[im1, j, 1:NK])
        local sumgi = (seqsum(@view gi[im1, j, 1:NK, 1]), seqsum(@view gi[im1, j, 1:NK, 2]))
        
        fn[i, j] = fn[i, j] - eg * (sumcxf + constv * sumuf) + gprinv * sumsif + constv * sumhxn
        chf[1, i, j] -= sumgi[1]
        chf[3, i, j] += sumgi[1]
        chf[4, i, j] -= 0.25e0 * sumgi[2]
        chf[5, i, j] += 0.25e0 * sumgi[2]
        chf[7, i, j] -= 0.25e0 * sumgi[2]
        chf[9, i, j] += 0.25e0 * sumgi[2]
        
        if (j == NJ)
          local sumvf = seqsum(@view vfbcn[i, 1:NK])
          fn[i, j] = fn[i, j] + eg * sumvf * kaphinv
        else
          local sumsjf = seqsum(@view sjfc[i, j, 1:NK])
          local sumcyf = seqsum(@view cyf[i, j, 1:NK])
          local sumvf = seqsum(@view vf[i, j, 1:NK])
          local sumhyn = seqsum(@view hyn[i, j, 1:NK])
          local sumgj = (seqsum(@view gj[i, j, 1:NK, 1]), seqsum(@view gj[i, j, 1:NK, 2]))
          fn[i, j] = fn[i, j] + eg * (sumcyf + constv *sumvf) - gprinv * sumsjf - constv * sumhyn
          chf[1, i, j] -=          sumgj[2]
          chf[2, i, j] += 0.25e0 * sumgj[1]
          chf[3, i, j] -= 0.25e0 * sumgj[1]
          chf[4, i, j] +=          sumgj[2]
          chf[6, i, j] += 0.25e0 * sumgj[1]
          chf[7, i, j] -= 0.25e0 * sumgj[1]
        end
        
        if (j == 1)
          local sumvf = seqsum(@view vfbcs[i, 1:NK])
          fn[i, j] = fn[i, j] - eg * sumvf * kaphinv
        else
          local sumsjf = seqsum(@view sjfc[i, j-1, 1:NK])
          local sumcyf = seqsum(@view cyf[i, j-1, 1:NK])
          local sumvf = seqsum(@view vf[i, j-1, 1:NK])
          local sumhyn = seqsum(@view hyn[i, j-1, 1:NK])
          local sumgj = (seqsum(@view gj[i, j-1, 1:NK, 1]), seqsum(@view gj[i, j-1, 1:NK, 2]))
          fn[i, j] = fn[i, j] - eg * (sumcyf + constv *sumvf) + gprinv * sumsjf + constv * sumhyn
          chf[1, i, j] -=          sumgj[2]
          chf[2, i, j] -= 0.25e0 * sumgj[1]
          chf[3, i, j] += 0.25e0 * sumgj[1]
          chf[5, i, j] +=          sumgj[2]
          chf[8, i, j] -= 0.25e0 * sumgj[1]
          chf[9, i, j] += 0.25e0 * sumgj[1]
        end
      end
    end
  end

end
