# Multi-colour ordering (same as linerelax_periodicew.f90, hsolve.f90, mprove.f90): the stencils
# couple point/column (i,j) to (i+-1,j+-1), so the points are split into colours by the parity of i
# and of j and the colours are swept one after the other. Points of one colour are independent (can
# be updated in parallel, in any order). x is periodic: for odd n, i=n neighbours both i=1 and
# i=n-1 and gets a third i-colour.
ncolours_i(n) = n == 1 ? 1 : (isodd(n) ? 3 : 2)
function colour_irange(n, ci, nci)
  ci == 3 && return n:n
  return ci:2:(nci == 3 ? n - 1 : n)
end

# One colour of the line relaxation: work item (a, b) solves the column (i, j) of that colour.
# rh, subd, supd, dia: per-column scratch (NK, NI, NJ); err: set to 1 if dgtsl fails.
@kernel function k_linerelax_colour!(p, cp, fn, rh, subd, supd, dia, err, nxm, nzm, ib, is, cj, lo)
  a, b = @index(Global, NTuple)
  @inbounds begin
    local i = ib + (a - 1) * is
    local j = cj + (b - 1) * 2
    local im1 = i - 1
    local ip1 = i + 1
    if (i == 1)
      im1 = nxm
    end
    if (i == nxm)
      ip1 = 1
    end
    for k in 1:nzm
      rh[k, i, j] = -(cp[2, i, j, k] *    p[ip1, j, k]
                      + cp[3, i, j, k] *  p[i, j+1, k]
                      + cp[4, i, j, k] *  p[im1, j, k]
                      + cp[5, i, j, k] *  p[i, j-1, k]
                      + cp[8, i, j, k] *  p[im1, j+1, k]
                      + cp[9, i, j, k] *  p[im1, j-1, k]
                      + cp[10, i, j, k] * p[ip1, j-1, k]
                      + cp[11, i, j, k] * p[ip1, j+1, k]
                      + cp[12, i, j, k] * p[im1, j, k-1]
                      + cp[13, i, j, k] * p[ip1, j, k-1]
                      + cp[14, i, j, k] * p[ip1, j, k+1]
                      + cp[15, i, j, k] * p[im1, j, k+1]
                      + cp[16, i, j, k] * p[i, j-1, k-1]
                      + cp[17, i, j, k] * p[i, j+1, k-1]
                      + cp[18, i, j, k] * p[i, j+1, k+1]
                      + cp[19, i, j, k] * p[i, j-1, k+1]
                      - fn[i, j, k])
      if (k != 1)
        subd[k, i, j] = cp[7, i, j, k]
      end
      if (k != nzm)
        supd[k, i, j] = cp[6, i, j, k]
      end
      dia[k, i, j] = cp[1, i, j, k]
    end
    rh[1, i, j] -= cp[7, i, j, 1] * p[i, j, 0]
    rh[nzm, i, j] -= cp[6, i, j, nzm] * p[i, j, nzm+1]

    local info = dgtsl(nzm, view(subd, :, i, j), view(dia, :, i, j), view(supd, :, i, j), view(rh, :, i, j))
    if (info != 0)
      err[1] = 1
    end
    for k in 1:nzm
      p[i, j, k] = rh[k, i, j]
    end
  end
end

function linerelax(nxm, nym, nzm, cp, p, fn, S=DEV)
  local rh = work(S, :linerelax_rh, rc_kind, NK, NI, NJ)
  local subd = work(S, :linerelax_subd, rc_kind, NK, NI, NJ)
  local supd = work(S, :linerelax_supd, rc_kind, NK, NI, NJ)
  local dia = work(S, :linerelax_dia, rc_kind, NK, NI, NJ)
  local err = work(S, :linerelax_err, Int, 1)
  local nci = ncolours_i(nxm)
  for cj in 1:2, ci in 1:nci
    local ir = colour_irange(nxm, ci, nci)
    launch!(S, k_linerelax_colour!, (1:length(ir), 1:length(cj:2:nym)), p, cp, fn, rh, subd, supd, dia, err,
            nxm, nzm, first(ir), step(ir), cj)
  end
  if maximum(err) != 0
    println("error in linerelax")
    exit(1)
  end
end
