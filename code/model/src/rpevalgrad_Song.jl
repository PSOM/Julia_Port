# Baroclinic pressure gradient, Song's scheme.

# rp: vertical integral of the density anomaly, from the surface down; one work item per (i,j) column
@kernel function k_rpSong_rp!(rp, zc, zf, rho, vconst, lo)
   i, j = @index(Global, NTuple) .+ lo
   @inbounds begin
      #     for k=NK (firt for the toppmost layer of cells)
      #     z is the dimensional z and is therefore multiplied by DL
      local k = NK
      local z = DL * zc[i, j, k]
      local zt = DL * zf[i, j, k]
      local zb = DL * zf[i, j, k-1]
      local rg = (rho[i, j, k] - R0) * vconst
      rp[i, j, k] = rg * (zt - z)
      local temp = rg * (z - zb)
      #     for the rest of the column
      #     remember that findz will not work for the k=0 cell where sig is -v
      #     We will simply fill rp at k=0 using extrapolation - since this
      #     is what we use to fill s and T in any case.
      for k in NK-1:-1:1
         z = DL * zc[i, j, k]
         zt = zb
         zb = DL * zf[i, j, k-1]
         rg = (rho[i, j, k] - R0) * vconst
         rp[i, j, k] = rg * (zt - z) + temp + rp[i, j, k+1]
         temp = rg * (z - zb)
      end
   end
end

#     Boundary conditions (first in j, then periodic in i, which also fills the corners)
@kernel function k_rpSong_rp_bcj!(rp, lo)
   i, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      rp[i, 0, k] = rp[i, 1, k]
      rp[i, NJ+1, k] = rp[i, NJ, k]
   end
end
@kernel function k_rpSong_rp_bci!(rp, lo)
   j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      rp[NI+1, j, k] = rp[1, j, k]
      rp[0, j, k] = rp[NI, j, k]
   end
end

# y-direction: Jy = Jy*dzsig - Jy2*dzeta at the j-faces j (Jy = 0 at the solid boundaries j = 0, NJ)
@kernel function k_rpSong_Jy!(Jy, rho, zc, lo)
   i, j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      if (j == 0 || j == NJ)
         Jy[i, j, k] = 0e0
      else
         local jy = 0e0
         local dzeta = 0e0
         local jy2 = 0e0
         local dzsig = 0e0
         if (k == 0)
            jy = 0.5e0 * (rho[i, j+1, k+1] + rho[i, j+1, k+2] - (rho[i, j, k+1] + rho[i, j, k+2]))
            dzeta = 0.5e0 * (zc[i, j+1, k+1] + zc[i, j+1, k+2] - (zc[i, j, k+1] + zc[i, j, k+2])) * DL
            jy2 = 0.5e0 * (rho[i, j, k+2] + rho[i, j+1, k+2] - (rho[i, j, k+1] + rho[i, j+1, k+1]))
            dzsig = 0.5e0 * (zc[i, j, 2] + zc[i, j+1, 2] - (zc[i, j, 1] + zc[i, j+1, 1])) * DL
         elseif (k == NK)
            jy = 0.5e0 * (rho[i,j+1, k] + rho[i, j+1, k-1] - (rho[i, j, k] + rho[i, j, k-1]))
            dzeta = 0.5e0 * (zc[i,j+1, k] + zc[i, j+1, k-1] - (zc[i, j, k] + zc[i, j, k-1])) * DL
            jy2 = 0.5e0 * (rho[i, j, NK] + rho[i, j+1, NK] - (rho[i, j, NK-1] + rho[i, j+1, NK-1]))
            dzsig = 0.5e0 * (zc[i, j, NK] + zc[i, j+1, NK] - (zc[i, j, NK-1] + zc[i, j+1, NK-1])) * DL
         else
            jy = 0.5e0 * (rho[i, j+1, k] + rho[i, j+1, k+1] - (rho[i, j, k] + rho[i, j, k+1]))
            dzeta = 0.5e0 * (zc[i, j+1, k] + zc[i, j+1, k+1] - (zc[i, j, k] + zc[i, j, k+1])) * DL
            jy2 = 0.5e0 * (rho[i, j, k+1] + rho[i, j+1, k+1] - (rho[i, j, k] + rho[i, j+1, k]))
            dzsig = 0.5e0 * (zc[i, j, k+1] + zc[i, j+1, k+1] - (zc[i, j, k] + zc[i, j+1, k])) * DL
         end
         jy *= dzsig
         jy2 *= dzeta
         jy -= jy2
         Jy[i, j, k] = jy
      end
   end
end

# grpjfc (j-faces j = 0:NJ) and drpy (centres j >= 1): vertical sums from the top; one work item per column
@kernel function k_rpSong_y!(grpjfc, drpy, Jy, gj, vy, vconst, lo)
   i, j = @index(Global, NTuple) .+ lo
   @inbounds begin
      if (j == 0 || j == NJ)
         for k in 1:NK
            grpjfc[i, j, k] = 0e0
         end
      else
         local k = NK
         local Jsum = Jy[i, j, k] * 0.5e0
         grpjfc[i, j, NK] = Jsum * vconst * gj[i, j, k, 2]
         for k in NK-1:-1:1
            Jsum += Jy[i, j, k]
            grpjfc[i, j, k] = Jsum * vconst * gj[i, j, k, 2]
         end
      end
      if (j >= 1)
         local k = NK
         drpy[i,j,k] = vconst * vy[i, j] * 0.5e0 * (Jy[i, j, NK] + Jy[i, j-1, NK]) * 0.5e0
         for k in NK-1:-1:1
            drpy[i,j,k] = drpy[i,j,k+1] + vconst * vy[i, j] * 0.5e0 * (Jy[i, j, k] + Jy[i, j-1, k])
         end
      end
   end
end

# x-direction (periodic): Jx = Jx*dzsig_x - Jx2*dzxi at the i-faces i = 1:NI (face NI uses column 1)
@kernel function k_rpSong_Jx!(Jx, rho, zc, lo)
   i, j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      local ip = (i == NI) ? 1 : i + 1
      local jx = 0e0
      local dzxi = 0e0
      local jx2 = 0e0
      local dzsig_x = 0e0
      if (k == 0)
         jx = rho[ip, j, k+1] - rho[i, j, k+1]
         dzxi = (zc[ip, j, k+1] - zc[i, j, k+1]) * DL
         # Fortran writes Jx2(j,0) here (index typo); the k=0 level is never used, so write the
         # i-indexed entry instead (avoids a BoundsError when NJ > NI)
         jx2 = 0.5e0 * (rho[i, j, k+2] + rho[ip, j, k+2] - (rho[i, j, k+1] + rho[ip, j, k+1]))
         dzsig_x = 0.5e0 * (zc[i, j, k+2] + zc[ip, j, k+2] - (zc[i, j, k+1] + zc[ip, j, k+1])) * DL
      elseif (k == NK)
         jx = rho[ip, j, k] - rho[i, j, k]
         dzxi = (zc[ip, j, k] - zc[i, j, k]) * DL
         jx2 = 0.5e0 * (rho[i, j, NK] + rho[ip, j, NK] - (rho[i, j, NK-1] + rho[ip, j, NK-1]))
         dzsig_x = 0.5e0 * (zc[i, j, NK] + zc[ip, j, NK] - (zc[i, j, NK-1] + zc[ip, j, NK-1])) * DL
      else
         jx = 0.5e0 * (rho[ip, j, k] + rho[ip, j, k+1] - (rho[i, j, k] + rho[i, j, k+1]))
         dzxi = 0.5e0 * (zc[ip, j, k] + zc[ip, j, k+1] - (zc[i, j, k] + zc[i, j, k+1])) * DL
         jx2 = 0.5e0 * (rho[i, j, k+1] + rho[ip, j, k+1] - (rho[i, j, k] + rho[ip, j, k]))
         dzsig_x = 0.5e0 * (zc[i, j, k+1] + zc[ip, j, k+1] - (zc[i, j, k] + zc[ip, j, k])) * DL
      end
      jx *= dzsig_x
      jx2 *= dzxi
      jx -= jx2
      Jx[i, j, k] = jx
   end
end

# grpifc and drpx: vertical sums from the top; one work item per column. Jx at face 0 is Jx at face NI.
@kernel function k_rpSong_x!(grpifc, drpx, Jx, gi, ux, vconst, lo)
   i, j = @index(Global, NTuple) .+ lo
   @inbounds begin
      local im = (i == 1) ? NI : i - 1
      local k = NK
      local Jsum = Jx[i, j, k] * 0.5e0
      grpifc[i, j, NK] = Jsum * vconst * gi[i, j, k, 1]
      for k in NK-1:-1:1
         Jsum += Jx[i, j, k]
         grpifc[i, j, k] = Jsum * vconst * gi[i, j, k, 1]
      end
      k = NK
      drpx[i,j,k] = vconst * ux[i, j] * 0.5e0 * (Jx[i, j, NK] + Jx[im, j, NK]) * 0.5e0
      for k in NK-1:-1:1
         drpx[i,j,k] = drpx[i,j,k+1] + vconst * ux[i, j] * 0.5e0 * (Jx[i, j, k] + Jx[im, j, k])
      end
   end
end

@kernel function k_rpSong_grpifc_bc!(grpifc, lo)
   j, k = @index(Global, NTuple) .+ lo
   @inbounds grpifc[0, j, k] = grpifc[NI, j, k]
end

function rpevalgrad_Song(n, S=DEV)
   local Jy = work(S, :rpevalgrad_Song_Jy, rc_kind, 1:NI, 0:NJ, 0:NK)
   local Jx = work(S, :rpevalgrad_Song_Jx, rc_kind, 1:NI, 1:NJ, 0:NK)
   if (!rect)
      println("modify grpifc,grpjfc")
      exit(1)
   end
   #      Evaluate Jacobian at the edge-centers
   #      multiply press and press gradient by const
   local vconst = 10e0 * gpr / P1

   findzall(S)

   evalrho(S.rho, n, S)
   # ===================================================================================

   launch!(S, k_rpSong_rp!, (0:NI+1, 0:NJ+1), S.rp, S.zc, S.zf, S.rho, vconst)

   #     Boundary conditions
   launch!(S, k_rpSong_rp_bcj!, (1:NI, 1:NK), S.rp)
   launch!(S, k_rpSong_rp_bci!, (0:NJ+1, 1:NK), S.rp)

   # ===================================================================================
   # c     y-direction
   # c     -----------
   # c     sloping sigma surfaces
   # c     solid boundaries
   launch!(S, k_rpSong_Jy!, (1:NI, 0:NJ, 0:NK), Jy, S.rho, S.zc)
   launch!(S, k_rpSong_y!, (1:NI, 0:NJ), S.grpjfc, S.drpy, Jy, S.gj, S.vy, vconst)

   #   ===================================================================================
   # c     x- direction
   # c     ------------------------
   # c     sloping sigma surfaces
   # c     periodic boundaries
   launch!(S, k_rpSong_Jx!, (1:NI, 1:NJ, 0:NK), Jx, S.rho, S.zc)
   launch!(S, k_rpSong_x!, (1:NI, 1:NJ), S.grpifc, S.drpx, Jx, S.gi, S.ux, vconst)
   launch!(S, k_rpSong_grpifc_bc!, (1:NJ, 1:NK), S.grpifc)
end
