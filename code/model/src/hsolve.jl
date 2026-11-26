# Free-surface solve: 9-point elliptic equation for h, multi-colour SOR (as hsolve.f90), then
# mprove (iterative improvement of the solution).

# Vertical sums needed by hfill at the south (j=0) and north (j=NJ) faces, computed once per hsolve
# (sequential k=1..NK, same order as the Fortran hfill loops). Columns of hsums:
#   1 vfbcs  2 cyf(j=0)   3 sjfc(j=0)   4 hyn(j=0)   5 gj(j=0,1)   6 gj(j=0,2)
#   7 vfbcn  8 cyf(j=NJ)  9 sjfc(j=NJ) 10 hyn(j=NJ) 11 gj(j=NJ,1) 12 gj(j=NJ,2)
@kernel function k_hsolve_sums!(hsums, vfbcs, vfbcn, cyf, sjfc, hyn, gj, lo)
  i, = @index(Global, NTuple) .+ lo
  @inbounds begin
    local s1 = 0e0; local s2 = 0e0; local s3 = 0e0; local s4 = 0e0; local s5 = 0e0; local s6 = 0e0
    local s7 = 0e0; local s8 = 0e0; local s9 = 0e0; local s10 = 0e0; local s11 = 0e0; local s12 = 0e0
    for k in 1:NK
      s1 += vfbcs[i, k]; s2 += cyf[i, 0, k]; s3 += sjfc[i, 0, k]; s4 += hyn[i, 0, k]
      s5 += gj[i, 0, k, 1]; s6 += gj[i, 0, k, 2]
      s7 += vfbcn[i, k]; s8 += cyf[i, NJ, k]; s9 += sjfc[i, NJ, k]; s10 += hyn[i, NJ, k]
      s11 += gj[i, NJ, k, 1]; s12 += gj[i, NJ, k, 2]
    end
    hsums[i, 1] = s1; hsums[i, 2] = s2; hsums[i, 3] = s3; hsums[i, 4] = s4; hsums[i, 5] = s5; hsums[i, 6] = s6
    hsums[i, 7] = s7; hsums[i, 8] = s8; hsums[i, 9] = s9; hsums[i, 10] = s10; hsums[i, 11] = s11; hsums[i, 12] = s12
  end
end

function hsolve_sums!(hsums, S)
  launch!(S, k_hsolve_sums!, (1:NI,), hsums, S.vfbcs, S.vfbcn, S.cyf, S.sjfc, S.hyn, S.gj)
end

# One colour of the SOR sweep for the 9-point operator ch (also used by mprove): points
# (i,j) of one colour are independent (see colour_irange in linerelax_periodicew.jl).
@kernel function k_sor9_colour!(h, ch, rhs, rlx, ib, is, cj, lo)
  a, b = @index(Global, NTuple)
  @inbounds begin
    local i = ib + (a - 1) * is
    local j = cj + (b - 1) * 2
    local im1 = i == 1 ? NI : i - 1
    local ip1 = i == NI ? 1 : i + 1
    local hstar = ( ch[2, i, j] * h[ip1, j]
                  + ch[3, i, j] * h[im1, j]
                  + ch[4, i, j] * h[i, j+1]
                  + ch[5, i, j] * h[i, j-1]
                  + ch[6, i, j] * h[ip1, j+1]
                  + ch[7, i, j] * h[im1, j+1]
                  + ch[8, i, j] * h[ip1, j-1]
                  + ch[9, i, j] * h[im1, j-1]
                  - rhs[i, j]) / (-ch[1, i, j])
    h[i, j] = (1e0 - rlx) * h[i, j] + rlx * hstar
  end
end

"multi-colour SOR sweep of the 9-point equation ch*h = rhs (periodic in x)"
function sor9_sweep!(h, ch, rhs, rlx, S)
  local nci = ncolours_i(NI)
  for cj in 1:2, ci in 1:nci
    local ir = colour_irange(NI, ci, nci)
    launch!(S, k_sor9_colour!, (1:length(ir), 1:length(cj:2:NJ)), h, ch, rhs, rlx, first(ir), step(ir), cj)
  end
end

# |rhs - ch*h| at every point (residual of the 9-point equation)
@kernel function k_res9!(res, h, ch, rhs, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local im1 = i == 1 ? NI : i - 1
    local ip1 = i == NI ? 1 : i + 1
    res[i, j] = abs(rhs[i, j] - (
        ch[1, i, j] * h[i, j]
      + ch[2, i, j] * h[ip1, j]
      + ch[3, i, j] * h[im1, j]
      + ch[4, i, j] * h[i, j+1]
      + ch[5, i, j] * h[i, j-1]
      + ch[6, i, j] * h[ip1, j+1]
      + ch[7, i, j] * h[im1, j+1]
      + ch[8, i, j] * h[ip1, j-1]
      + ch[9, i, j] * h[im1, j-1]))
  end
end

"max |rhs - ch*h|"
function res9_max(h, ch, rhs, S)
  local res = work(S, :res9, rc_kind, NI, NJ)
  launch!(S, k_res9!, (1:NI, 1:NJ), res, h, ch, rhs)
  return maximum(res)
end

function hsolve(h, oldh, hdt, dtime, S=DEV)
  local ch = work(S, :hsolve_ch, rc_kind, 9, NI, NJ)
  local rhs = work(S, :hsolve_rhs, rc_kind, NI, NJ)
  local hsums = work(S, :hsolve_sums, rc_kind, NI, 12)

  local tol::rc_kind = 1e-12

  hsolve_sums!(hsums, S)
  # compute fine grid coefficients and rhs on fine grid
  chfine(dtime, ch, rhs, S)

  local rlx::rc_kind = 1.72e0

  for iter in 1:100
    sor9_sweep!(h, ch, rhs, rlx, S)

    for l in 1:3
      hfill(dtime, h, hsums, S)
    end

    local maxres = res9_max(h, ch, rhs, S)
    if (maxres > 3000)
      println("hsolve.jl: STOP. res too large, maxres=", maxres)
      exit(1)
    end
    if (maxres < tol)
      break
    end
  end

  mprove(h, ch, rhs, dtime, S)
end
