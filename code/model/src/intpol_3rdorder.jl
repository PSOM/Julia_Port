# Interpolation of the cell-centre cx, cy, cz to the faces (cxf, cyf, czf); 3rd order in the horizontal.

@kernel function k_intpol_cxf!(cxf, ux, uy, cx, cy, Jifc, lo)
   i, j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      local xa::rc_kind = 9e0 / 16e0
      local xb::rc_kind = -1e0 / 16e0
      local im1::Int = 0
      local ip1::Int = 0
      local ip2::Int = 0
      local i0::Int = 0
      if (i == 0)
         i0 = NI
         im1 = NI-1
         ip1 = i+1
         ip2 = i+2
      elseif (i == 1)
         i0 = i
         im1 = NI
         ip1 = i+1
         ip2 = i+2
      elseif (i == NI)
         i0 = i
         im1 = i-1
         ip1 = 1
         ip2 = 2
      elseif (i == NI-1)
         i0 = i
         im1 = i-1
         ip1 = i+1
         ip2 = 1
      else
         i0 = i
         im1 = i-1
         ip1 = i+1
         ip2 = i+2
      end
      cxf[i,j,k] = ( xa * (ux[i0, j] * cx[i0, j, k] + ux[ip1, j] * cx[ip1, j, k])
                     + xb * (ux[im1, j] * cx[im1, j, k] + ux[ip2, j] * cx[ip2, j, k])
                     + 0.5e0 * (uy[i0, j] * cy[i0, j, k] + uy[ip1, j] * cy[ip1, j, k])) * Jifc[i, j, k]
   end
end

@kernel function k_intpol_cxf_bc!(cxf, ufbcw, ufbce, lo)
   j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      cxf[0, j, k] = ufbcw[j, k]
      cxf[NI, j, k] = ufbce[j, k]
   end
end

@kernel function k_intpol_cyf!(cyf, vx, vy, cx, cy, Jjfc, lo)
   i, j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      local j0::Int = 0
      local jm1::Int = 0
      local jp1::Int = 0
      local jp2::Int = 0
      local xa::rc_kind = 0e0
      local xb::rc_kind = 0e0
      if (j == 1)
         j0 = j
         jm1 = NJ
         jp1 = j+1
         jp2 = j+2
         xa = 0.5
         xb = 0.0
      elseif (j == NJ-1)
         j0 = j
         jm1 = j-1
         jp1 = NJ
         jp2 = 1
         xa = 0.5
         xb = 0.0
      else
         j0 = j
         jm1 = j-1
         jp1 = j+1
         jp2 = j+2
         xa = 9e0/16e0
         xb = -1e0/16e0
      end
      cyf[i, j, k] = Jjfc[i, j, k] * (
         0.5e0 * (vx[i, j0] * cx[i, j0, k] + vx[i, jp1] * cx[i, jp1, k])
          + xa * (vy[i, j0] * cy[i, j0, k] + vy[i, jp1] * cy[i, jp1, k])
          + xb * (vy[i, jm1] * cy[i, jm1, k] + vy[i, jp2] * cy[i, jp2, k])
      )
   end
end

@kernel function k_intpol_cyf_bc!(cyf, vfbcn, vfbcs, lo)
   i, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      cyf[i, NJ, k] = vfbcn[i,k]
      cyf[i, 0, k] = vfbcs[i,k]
   end
end

@kernel function k_intpol_czf!(czf, Jac, wx, wy, wz, cx, cy, cz, lo)
   i, j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      local Jack::rc_kind = 0.5e0 * (Jac[i, j, k] + Jac[i, j, k+1])
      czf[i, j, k] = 0.5 * Jack * ( wx[i, j, k] * cx[i, j, k] + wx[i, j, k+1] * cx[i, j, k+1]
                                    + wy[i, j, k] * cy[i, j, k] + wy[i, j, k+1] * cy[i, j, k+1]
                              + EPS *(wz[i, j, k] * cz[i, j, k] + wz[i, j, k+1] * cz[i, j, k+1]))
   end
end

@kernel function k_intpol_czf_bc!(czf, wfbcb, Jac, wx, wy, wz, cx, cy, cz, lo)
   i, j = @index(Global, NTuple) .+ lo
   @inbounds begin
      czf[i, j, 0] = wfbcb[i,j]
      czf[i, j, NK] = Jac[i, j, NK] * (wx[i, j, NK] * cx[i, j, NK] + wy[i, j, NK] * cy[i, j, NK] + EPS * wz[i, j, NK] * cz[i, j, NK])
   end
end

function intpol(S=DEV)
   if (NI < 4 || NJ < 4)
      println("prob in intpol,insuff i pts")
      exit(1)
   end

   launch!(S, k_intpol_cxf!, (0:NI, 1:NJ, 1:NK), S.cxf, S.ux, S.uy, S.cx, S.cy, S.Jifc)

   if (!periodicew)
      launch!(S, k_intpol_cxf_bc!, (1:NJ, 1:NK), S.cxf, S.ufbcw, S.ufbce)
   end

   launch!(S, k_intpol_cyf!, (1:NI, 1:NJ-1, 1:NK), S.cyf, S.vx, S.vy, S.cx, S.cy, S.Jjfc)
   launch!(S, k_intpol_cyf_bc!, (1:NI, 1:NK), S.cyf, S.vfbcn, S.vfbcs)

   launch!(S, k_intpol_czf!, (1:NI, 1:NJ, 1:NK-1), S.czf, S.Jac, S.wx, S.wy, S.wz, S.cx, S.cy, S.cz)
   launch!(S, k_intpol_czf_bc!, (1:NI, 1:NJ), S.czf, S.wfbcb, S.Jac, S.wx, S.wy, S.wz, S.cx, S.cy, S.cz)
end
