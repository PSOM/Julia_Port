# hfill: h at the south/north ghost rows (j=0, NJ+1) from the boundary condition, then the
# periodic x halo. hsums: the vertical sums of hsolve_sums! (columns: see there).
#
# Each row is a sequential sweep in i, as in hfill.f90: h(i,0) uses the already updated h(i-1,0),
# so one work item per row (2 in total) does the whole row, which keeps the Fortran result exactly.
@kernel function k_hfill_rows!(hf, hsums, edt, cons, gkn, lo)
  r, = @index(Global, NTuple)          # r = 1: south row (j=0), r = 2: north row (j=NJ+1)
  @inbounds if r == 1
    for i in 1:NI
      local hxi = 0.25e0 * (hf[i+1, 0] + hf[i+1, 1] - hf[i-1, 0] - hf[i-1, 1])
      local gradh = (edt * (hsums[i, 2] - hsums[i, 1]) - hsums[i, 3] - cons * hsums[i, 4]) / gkn
      hf[i, 0] = hf[i, 1] + (hsums[i, 5] * hxi - gradh) / hsums[i, 6]
    end
  else
    for i in 1:NI
      local hxi = 0.25e0 * (hf[i+1, NJ] + hf[i+1, NJ+1] - hf[i-1, NJ] - hf[i-1, NJ+1])
      # north boundary: vfbcn (Fortran hfill.f90)
      local gradh = (edt * (hsums[i, 8] - hsums[i, 7]) - hsums[i, 9] - cons * hsums[i, 10]) / gkn
      hf[i, NJ+1] = hf[i, NJ] + (-hsums[i, 11] * hxi + gradh) / hsums[i, 12]
    end
  end
end

# periodic halo in x (after the rows: it copies the updated h(NI,0), h(1,NJ+1), ...)
@kernel function k_hfill_periodic!(hf, lo)
  j, = @index(Global, NTuple) .+ lo
  @inbounds begin
    hf[0, j] = hf[NI, j]
    hf[NI+1, j] = hf[1, j]
  end
end

function hfill(dtimel, hf, hsums, S=DEV)
  local edt = EPS / dtimel
  local cons = (1e0 - kappah) * gpr
  local gkn = gpr * kappah
  launch!(S, k_hfill_rows!, (1:2,), hf, hsums, edt, cons, gkn)
  launch!(S, k_hfill_periodic!, (0:NJ+1,), hf)
end
