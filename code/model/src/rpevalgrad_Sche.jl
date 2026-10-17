# Baroclinic pressure gradient, Shchepetkin & McWilliams (ROMS) scheme.
#
# Part I translates the PSOM variables into ROMS variables, Part II is the ROMS code, Part III translates
# back. ROMS indices: imin = istr = istrU = 1, imax = iend = NI, jmin = jstr = jstrV = 1, jmax = jend = NJ,
# N = NK. The work arrays FC, rx, dZx, dRx (2D per level k in ROMS) and dZ, dR (2D per row j) carry all
# three indices here, so that the levels (rows) are computed in parallel.

# Part I. Geometry and density anomaly, with the periodic x-boundaries of Hz and rhol.
@kernel function k_rpSche_geom!(z_r, z_w, Hz, rhol, zc, zf, rho, rho0, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    z_r[i, j, k] = DL * zc[i, j, k]
    z_w[i, j, k] = DL * zf[i, j, k]
    if (k == 0)
      z_w[i, j, k-1] = DL * zf[i, j, k-1]
    end
    # Boundary condition in x: Hz[0] = Hz[NI], Hz[NI+1] = Hz[1], same for rhol
    local ii = (i == 0) ? NI : ((i == NI + 1) ? 1 : i)
    if (1 <= k <= NK)
      Hz[i, j, k] = (DL * zf[ii, j, k] - DL * zf[ii, j, k-1])
    end
    rhol[i, j, k] = rho[ii, j, k] - rho0
  end
end

# Preliminary step (same for XI- and ETA-components); one work item per (i,j) column
@kernel function k_rpSche_P!(P_Sche, dZ, dR, z_r, z_w, rhol, epsil, g, GRho, HalfGRho, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local N = NK
    for k in 1:N-1
      dZ[i, j, k] = z_r[i, j, k+1] - z_r[i, j, k]
      dR[i, j, k] = rhol[i, j, k+1] - rhol[i, j, k]
    end
    dR[i, j, N] = dR[i, j, N-1]
    dR[i, j, 0] = dR[i, j, 1]
    dZ[i, j, N] = dZ[i, j, N-1]
    dZ[i, j, 0] = dZ[i, j, 1]
    for k in N:-1:1
      local cff = 2e0 * dZ[i, j, k] * dZ[i, j, k-1]
      dZ[i, j, k] = cff / (dZ[i, j, k] + dZ[i, j, k-1])

      local cfr = 2e0 * dR[i, j, k] * dR[i, j, k-1]
      if (cfr > epsil)
        dR[i, j, k] = cfr / (dR[i, j, k] + dR[i, j, k-1])
      else
        dR[i, j, k] = 0e0
      end
    end
    P_Sche[i, j, N] = 0e0 * g * z_w[i, j, N] + GRho * (rhol[i, j, N] + 0.5 * (rhol[i, j, N] - rhol[i, j, N-1]) * (z_w[i, j, N] - z_r[i, j, N]) / (z_r[i, j, N] - z_r[i, j, N-1])) * (z_w[i, j, N] - z_r[i, j, N])
    for k in N-1:-1:1
      P_Sche[i, j, k] = P_Sche[i, j, k+1] + HalfGRho * ((rhol[i, j, k+1] + rhol[i, j, k]) * (z_r[i, j, k+1] - z_r[i, j, k])
                                                        -
                                                        0.2e0 * ((dR[i, j, k+1] - dR[i, j, k]) * (z_r[i, j, k+1] - z_r[i, j, k] - (1e0 / 12e0) * (dZ[i, j, k+1] + dZ[i, j, k]))
                                                                 -
                                                                 (dZ[i, j, k+1] - dZ[i, j, k]) * (rhol[i, j, k+1] - rhol[i, j, k] - (1e0 / 12e0) * (dR[i, j, k+1] + dR[i, j, k]))))
    end
  end
end

# Caution:
# The line P_Sche(i,j,N)=... has been modified !
# The term g*z_w(i,j,N) has been set to 0 because PSOM only requires the baroclinic terms and not
#  the barotropic part linked with surface elevation (g.R0.h).

# XI-component: FC, rx at i = 0:NI+1, j = 1:NJ.
# This part has been included to take into account the boundary conditions (added by JBG, 20120425):
# FC[1] = z_r[1] - z_r[NI], FC[0] = FC[NI], FC[NI+1] = FC[1]; same for rx.
@kernel function k_rpSche_xi_fc!(FC, rx, z_r, rhol, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local ii = (i == 0) ? NI : ((i == NI + 1) ? 1 : i)
    if (ii == 1)
      FC[i, j, k] = z_r[1, j, k] - z_r[NI, j, k]
      rx[i, j, k] = rhol[1, j, k] - rhol[NI, j, k]
    else
      FC[i, j, k] = z_r[ii, j, k] - z_r[ii-1, j, k]
      rx[i, j, k] = rhol[ii, j, k] - rhol[ii-1, j, k]
    end
  end
end

@kernel function k_rpSche_xi_d!(dZx, dRx, FC, rx, epsil, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local cff = 2e0 * FC[i, j, k] * FC[i+1, j, k]
    if (cff > epsil)
      dZx[i, j, k] = cff / (FC[i, j, k] + FC[i+1, j, k])
    else
      dZx[i, j, k] = 0e0
    end

    local cfr = 2e0 * rx[i, j, k] * rx[i+1, j, k]
    if (cfr > epsil)
      dRx[i, j, k] = cfr / (rx[i, j, k] + rx[i+1, j, k])
    else
      dRx[i, j, k] = 0e0
    end
  end
end

# dn_u = dx
@kernel function k_rpSche_ru!(ru_Sche, Hz, P_Sche, rhol, z_r, dZx, dRx, dn_u, HalfGRho, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    ru_Sche[i, j, k] = 0.5e0 * (Hz[i, j, k] + Hz[i-1, j, k]) * dn_u * (
                             (P_Sche[i-1, j, k] - P_Sche[i, j, k]) - HalfGRho * (
                               (rhol[i, j, k] + rhol[i-1, j, k]) * (z_r[i, j, k] - z_r[i-1, j, k])
                               -
                               0.2e0 * ((dRx[i, j, k] - dRx[i-1, j, k]) * (z_r[i, j, k] - z_r[i-1, j, k]
                                                                         -
                                                                         (1e0 / 12e0) * (dZx[i, j, k] + dZx[i-1, j, k]))
                                        -
                                        (dZx[i, j, k] - dZx[i-1, j, k]) * (rhol[i, j, k] - rhol[i-1, j, k]
                                                                         -
                                                                         (1e0 / 12e0) * (dRx[i, j, k] + dRx[i-1, j, k]))
                               )
                             )
                           )
  end
end

# ETA-component: FC, rx at i = 1:NI, j = 0:NJ+1.
# This part has been altered to take into account the boundary conditions (modified by JBG, 20120425):
# FC[jmin] = FC[jmin+1], FC[jmin-1] = FC[jmin], FC[jmax] = FC[jmax-1], FC[jmax+1] = FC[jmax]; same for rx.
@kernel function k_rpSche_eta_fc!(FC, rx, z_r, rhol, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local jj = (j <= 1) ? 2 : ((j >= NJ) ? NJ - 1 : j)
    FC[i, j, k] = (z_r[i, jj, k] - z_r[i, jj-1, k])
    rx[i, j, k] = (rhol[i, jj, k] - rhol[i, jj-1, k])
  end
end

@kernel function k_rpSche_eta_d!(dZx, dRx, FC, rx, epsil, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local cff = 2e0 * FC[i, j, k] * FC[i, j+1, k]
    if (cff > epsil)
      dZx[i, j, k] = cff / (FC[i, j, k] + FC[i, j+1, k])
    else
      dZx[i, j, k] = 0e0
    end

    local cfr = 2e0 * rx[i, j, k] * rx[i, j+1, k]
    if (cfr > epsil)
      dRx[i, j, k] = cfr / (rx[i, j, k] + rx[i, j+1, k])
    else
      dRx[i, j, k] = 0e0
    end
  end
end

# dm_v = dy
@kernel function k_rpSche_rv!(rv_Sche, Hz, P_Sche, rhol, z_r, dZx, dRx, dm_v, HalfGRho, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    rv_Sche[i, j, k] = 0.5e0 * (Hz[i, j, k] + Hz[i, j-1, k]) * dm_v * (
                             (P_Sche[i, j-1, k] - P_Sche[i, j, k]) - HalfGRho * (
                               (rhol[i, j, k] + rhol[i, j-1, k]) * (z_r[i, j, k] - z_r[i, j-1, k]) - 0.2e0 * (
                                 (dRx[i, j, k] - dRx[i, j-1, k]) * (z_r[i, j, k] - z_r[i, j-1, k] - (1e0 / 12e0) * (dZx[i, j, k] + dZx[i, j-1, k]))
                                 -
                                 (dZx[i, j, k] - dZx[i, j-1, k]) * (rhol[i, j, k] - rhol[i, j-1, k] - (1e0 / 12e0) * (dRx[i, j, k] + dRx[i, j-1, k]))
                               )
                             )
                           )
  end
end

# nondimensionalization (whole arrays, as in ru_Sche .*= -adim_Sche)
@kernel function k_rpSche_scale!(ru_Sche, rv_Sche, fac, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    ru_Sche[i, j, k] *= fac
    rv_Sche[i, j, k] *= fac
  end
end

# ru2_Sche, rv2_Sche: shift by one point and boundary conditions (-999 where not defined)
@kernel function k_rpSche_shift!(ru2_Sche, rv2_Sche, ru_Sche, rv_Sche, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    if (i == 0 || i == NI)
      ru2_Sche[i, j, k] = ru_Sche[1, j, k]
    elseif (1 <= i <= NI-1 && 1 <= j <= NJ)
      ru2_Sche[i, j, k] = ru_Sche[i+1, j, k]
    else
      ru2_Sche[i, j, k] = -999e0
    end
    if (j == 0 || j == NJ)
      rv2_Sche[i, j, k] = 0e0
    elseif (1 <= j <= NJ-1 && 1 <= i <= NI)
      rv2_Sche[i, j, k] = rv_Sche[i, j+1, k]
    else
      rv2_Sche[i, j, k] = -999e0
    end
  end
end

# drpx, drpy from grpifc, grpjfc (the same relation as in the Song scheme)
@kernel function k_rpSche_drp!(ru3_Sche, rv3_Sche, ru4_Sche, rv4_Sche, ru2_Sche, rv2_Sche, gi, gj, ux, uy, vx, vy, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local ru3 = 0.5e0 * (ru2_Sche[i, j, k] / gi[i, j, k, 1] + ru2_Sche[i-1, j, k] / gi[i-1, j, k, 1])
    local rv3 = 0.5e0 * (rv2_Sche[i, j, k] / gj[i, j, k, 2] + rv2_Sche[i, j-1, k] / gj[i, j-1, k, 2])
    ru3_Sche[i, j, k] = ru3
    rv3_Sche[i, j, k] = rv3
    ru4_Sche[i, j, k] = (ru3 * ux[i, j] + rv3 * vx[i, j])
    rv4_Sche[i, j, k] = (ru3 * uy[i, j] + rv3 * vy[i, j])
  end
end

function rpevalgrad_Sche(nl::Int, S=DEV)
  local dZ = work(S, :rpevalgrad_Sche_dZ, rc_kind, 0:NI, 0:NJ, 0:NK)
  local dR = work(S, :rpevalgrad_Sche_dR, rc_kind, 0:NI, 0:NJ, 0:NK)
  local FC = work(S, :rpevalgrad_Sche_FC, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)
  local dZx = work(S, :rpevalgrad_Sche_dZx, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)
  local dRx = work(S, :rpevalgrad_Sche_dRx, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)
  local rx = work(S, :rpevalgrad_Sche_rx, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)
  local rhol = work(S, :rpevalgrad_Sche_rhol, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)
  local z_r = work(S, :rpevalgrad_Sche_z_r, rc_kind, 0:NI+1, 0:NJ+1, 0:NK+1)
  local Hz = work(S, :rpevalgrad_Sche_Hz, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)
  local z_w = work(S, :rpevalgrad_Sche_z_w, rc_kind, 0:NI+1, 0:NJ+1, -1:NK+1)
  local P_Sche = work(S, :rpevalgrad_Sche_P_Sche, rc_kind, 0:NI+1, 0:NJ+1, 1:NK)

  local epsil = 0e0

  # ====================================================================================
  # -----------------------------------------------------------------
  # Part I: The variables of PSOM are translated into ROMS variables.
  # -----------------------------------------------------------------

  # --------
  # Geometry
  local dn_u = dx
  local dm_v = dy

  # ----------
  # Geophysics
  local g = gpr * 10e0
  local rho0 = R0

  evalrho(S.rho, nl, S)

  launch!(S, k_rpSche_geom!, (0:NI+1, 0:NJ+1, 0:NK+1), z_r, z_w, Hz, rhol, S.zc, S.zf, S.rho, rho0)

  # ====================================================================================
  # -----------------------------------------------------------------
  # Part II: The code.
  # -----------------------------------------------------------------
  # The cpp tests have been removes.
  # Except from explicitely designed ligns, nothing has been changed.

  local GRho = g / rho0
  local HalfGRho = 0.5e0 * GRho

  launch!(S, k_rpSche_P!, (0:NI, 0:NJ), P_Sche, dZ, dR, z_r, z_w, rhol, epsil, g, GRho, HalfGRho)

  # Compute XI-component of pressure gradient term:
  launch!(S, k_rpSche_xi_fc!, (0:NI+1, 1:NJ, 1:NK), FC, rx, z_r, rhol)
  launch!(S, k_rpSche_xi_d!, (0:NI, 1:NJ, 1:NK), dZx, dRx, FC, rx, epsil)
  launch!(S, k_rpSche_ru!, (1:NI, 1:NJ, 1:NK), S.ru_Sche, Hz, P_Sche, rhol, z_r, dZx, dRx, dn_u, HalfGRho)

  # ETA-component of pressure gradient term:
  launch!(S, k_rpSche_eta_fc!, (1:NI, 0:NJ+1, 1:NK), FC, rx, z_r, rhol)
  launch!(S, k_rpSche_eta_d!, (1:NI, 0:NJ, 1:NK), dZx, dRx, FC, rx, epsil)
  launch!(S, k_rpSche_rv!, (1:NI, 1:NJ, 1:NK), S.rv_Sche, Hz, P_Sche, rhol, z_r, dZx, dRx, dm_v, HalfGRho)

  # ====================================================================================
  # -----------------------------------------------------------------
  # Part III: The variables of ROMS are now translated into PSOM variables.
  # -----------------------------------------------------------------

  #--------------------------------
  # ru_Sche and rv_Sche are the equivalent to grpifc and grpjfc but
  # - are dimensionalized
  # - have a shift of one grid point.
  # - the boundary conditions will have to be prescribed.

  #-----------------------
  #- nondimensionalization
  local adim_Sche = 1e0 / (P1 / R0 * dx * dy)
  launch!(S, k_rpSche_scale!, (0:NI+1, 0:NJ+1, 1:NK), S.ru_Sche, S.rv_Sche, -adim_Sche)

  #- shift and boundary conditions
  launch!(S, k_rpSche_shift!, (0:NI+1, 0:NJ+1, 1:NK), S.ru2_Sche, S.rv2_Sche, S.ru_Sche, S.rv_Sche)

  #--------------------------------
  # ru2_Sche and rv2_Sche are now totally equivalent to grpifc and grpjfc.
  # Now, we have to compute drpx and drpy.
  # The relation between grpifc/grpjfc and drpx/drpy is exactly the same as the one in the Song scheme.
  launch!(S, k_rpSche_drp!, (1:NI, 1:NJ, 1:NK), S.ru3_Sche, S.rv3_Sche, S.ru4_Sche, S.rv4_Sche,
          S.ru2_Sche, S.rv2_Sche, S.gi, S.gj, S.ux, S.uy, S.vx, S.vy)

  #---------------------------------
  # ru2_Sche and rv2_Sche are the grpifc and grpjfc
  # ru4_Sche and rv4_Sche are the drpx and drpy
  #---------------------------------
  #---------------------------------


  #---------------------------------
  # To illustrate how the code works on a simple case, 
  # Let us set:
  #  * rho=R0+ constant
  #  * h varies in y.
  #
  # The pressure terms can be decomposed into:
  #
  # (              Eq. P.2) only pressure terms from NK-1 to 1:                         0           0.737      0.737      0.737    ...
  # (part of Eq. P.1 + P.2) pressure terms from NK-1 to 1 + the rho term in NK face:    0.737       1.474      1.474      1.474    ...
  # (part of Eq. P.1 + P.2) pressure terms from NK-1 to 1 + the g.z_w term:           -55.75      -55.02     -55.02     -55.02     ...
  # (        Eq. P.1 + P.2) all pressure terms:				            -55.01      -54.28     -54.28     -54.28     ...
  #                                                                                                        
  # Then, a local additional term is added:                                                               
  #                                                                                                        
  # (part of Eq. R.1)       only this term:				 	      0.737       0.         0.         0.       ...
  #                                                                                                        
  # (Eq. P.1, P.2 and R.1) In the end, the addition of all these terms gives:         -54.276     -54.285    -54.285   -54.285     ...
  #---------------------------------




end
