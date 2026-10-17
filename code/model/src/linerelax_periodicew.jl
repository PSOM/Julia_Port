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

function linerelax_column(i, j, nxm, nzm, cp, p, fn, rh, subd, supd, dia)
  local im1 = i - 1
  local ip1 = i + 1
  if (i == 1)
    im1 = nxm
  end
  if (i == nxm)
    ip1 = 1
  end
  for k in 1:nzm
    rh[k] = -(cp[2, i, j, k] *    p[ip1, j, k]
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
      subd[k] = cp[7, i, j, k]
    end
    if (k != nzm)
      supd[k] = cp[6, i, j, k]
    end
    dia[k] = cp[1, i, j, k]
  end
  rh[1] -= cp[7, i, j, 1] * p[i, j, 0]
  rh[nzm] -= cp[6, i, j, nzm] * p[i, j, nzm+1]

  local info = dgtsl(nzm, subd, dia, supd, rh)
  if (info != 0)
    println("error in linerelax")
    exit(1)
  end
  for k in 1:nzm
    p[i, j, k] = rh[k]
  end
end

function linerelax(nxm, nym, nzm, cp, p, fn)
  local rh = zeros(rc_kind, NK)
  local subd = zeros(rc_kind, NK)
  local supd = zeros(rc_kind, NK)
  local dia = zeros(rc_kind, NK)
  local nci = ncolours_i(nxm)
  for cj in 1:2, ci in 1:nci
    for j in cj:2:nym, i in colour_irange(nxm, ci, nci)
      linerelax_column(i, j, nxm, nzm, cp, p, fn, rh, subd, supd, dia)
    end
  end
end
