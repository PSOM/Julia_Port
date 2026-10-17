# mgpfill: pressure at the points outside the impermeable boundaries (k=0, j=0, j=NJ+1), from the
# boundary condition v_normal = 0 (central differences for the boundary pressure gradient), using
# uf, vf, wf at n=0 as the boundary values. Then k=NK+1 (p = -p) and the periodic x halo.
# In this version the dpdsig term is left out at the vertical faces, because g13, g23 are larger
# than g11, g12. If the boundary conditions change with time, this routine must be changed.
#
# As in mgpfill.f90, the faces are filled in place in a fixed order (i outer, then j or k): a point
# uses its already-updated neighbours at i-1 and at j-1 (or k-1). To run this in parallel without
# changing the result, each face is swept as a wavefront over anti-diagonals (i + j or i + k = const),
# whose points are independent; one workgroup does the whole face, with a barrier per diagonal. The
# two bottom edge rows (j=0 and j=NJ+1 at k=0) are 1-D sequential sweeps in i (one work item each).

const MGPFILL_GROUPSIZE = 256

# face k=0, i=1..NI, j=1..NJ
@kernel function k_mgpfill_bottom!(pf, gqk, skfc, czf, wfbcb, edt)
  tid = @index(Local, Linear)
  nt = @uniform @groupsize()[1]
  for d in 2:NI+NJ
    local ilo = max(1, d - NJ)
    local ihi = min(NI, d - 1)
    local i = ilo + tid - 1
    while i <= ihi
      local j = d - i
      @inbounds begin
        local k = 0
        local gin = 1e0 / gqk[i, j, k, 3]
        # note that quantities are not defined at i=0,NI+1,j=0,NJ+1, and so we revert to filling
        # pf inside the domain only.
        local dpdxi = 0.25 * (pf[i+1, j, k+1] + pf[i+1, j, k] - pf[i-1, j, k+1] - pf[i-1, j, k])
        local dpdeta = 0.25 * (pf[i, j+1, k+1] + pf[i, j+1, k] - pf[i, j-1, k+1] - pf[i, j-1, k])
        local psig = gin * (-skfc[i, j, k] - gqk[i, j, k, 1] * dpdxi - gqk[i, j, k, 2] * dpdeta + edt * (czf[i, j, k] - wfbcb[i, j]))
        # since : wf(i,j,k)= czf(i,j,k) -dte*delta*(pz +skfc(i,j,k))
        pf[i, j, 0] = pf[i, j, 1] - psig
      end
      i += nt
    end
    @synchronize
  end
end

# bottom edges (k=0) at j=0 (r=1) and j=NJ+1 (r=2): sequential in i
@kernel function k_mgpfill_edges!(pf, gqk, skfc, lo)
  r, = @index(Global, NTuple)
  @inbounds begin
    local k = 0
    if r == 1
      local j = 0
      for i in 1:NI
        local gin = 1e0 / gqk[i, j, 0, 3]
        local dpdxi = 0.25 * (pf[i+1, j, k+1] + pf[i+1, j, k] - pf[i-1, j, k+1] - pf[i-1, j, k])
        local dpdeta = 0.5 * (pf[i, j+1, k+1] + pf[i, j+1, k] - pf[i, j, k+1] - pf[i, j, k])
        local psig = gin * (-skfc[i, j, k] - gqk[i, j, k, 1] * dpdxi - gqk[i, j, k, 2] * dpdeta)
        pf[i, j, 0] = pf[i, j, 1] - psig
      end
    else
      local j = NJ + 1
      for i in 1:NI
        local gin = 1e0 / gqk[i, j, 0, 3]
        local dpdxi = 0.25 * (pf[i+1, j, k+1] + pf[i+1, j, k] - pf[i-1, j, k+1] - pf[i-1, j, k])
        local dpdeta = 0.5 * (pf[i, j, k+1] + pf[i, j, k] - pf[i, j-1, k+1] - pf[i, j-1, k])
        local psig = gin * (-skfc[i, j, k] - gqk[i, j, k, 1] * dpdxi - gqk[i, j, k, 2] * dpdeta)
        pf[i, j, 0] = pf[i, j, 1] - psig
      end
    end
  end
end

# face k=NK+1: p = -p (all i=1..NI, j=0..NJ+1)
@kernel function k_mgpfill_top!(pf, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds pf[i, j, NK+1] = -pf[i, j, NK]
end

# faces j=0 (south) and j=NJ+1 (north), i=1..NI, k=1..NK, wavefront over i + k
@kernel function k_mgpfill_south!(pf, gqj, gqj3, cyf, vfbcs, sjfc, edt)
  tid = @index(Local, Linear)
  nt = @uniform @groupsize()[1]
  for d in 2:NI+NK
    local ilo = max(1, d - NK)
    local ihi = min(NI, d - 1)
    local i = ilo + tid - 1
    while i <= ihi
      local k = d - i
      @inbounds begin
        local j = 0
        local gin = 1e0 / gqj[i, j, k, 2]
        local dpdxi = 0.25 * (pf[i+1, j+1, k] + pf[i+1, j, k] - pf[i-1, j+1, k] - pf[i-1, j, k])
        local dpdsig = 0.25 * (pf[i, j+1, k+1] + pf[i, j, k+1] - pf[i, j+1, k-1] - pf[i, j, k-1])
        local peta = gin * (-gqj[i, j, k, 1] * dpdxi - gqj3[i, j, k] * dpdsig + (edt * (cyf[i, j, k] - vfbcs[i, k]) - sjfc[i, j, k]))
        pf[i, 0, k] = pf[i, 1, k] - peta
      end
      i += nt
    end
    @synchronize
  end
end

@kernel function k_mgpfill_north!(pf, gqj, gqj3, cyf, vfbcn, sjfc, edt)
  tid = @index(Local, Linear)
  nt = @uniform @groupsize()[1]
  for d in 2:NI+NK
    local ilo = max(1, d - NK)
    local ihi = min(NI, d - 1)
    local i = ilo + tid - 1
    while i <= ihi
      local k = d - i
      @inbounds begin
        local j = NJ
        local gin = 1e0 / gqj[i, j, k, 2]
        local dpdxi = 0.25 * (pf[i+1, j+1, k] + pf[i+1, j, k] - pf[i-1, j+1, k] - pf[i-1, j, k])
        local dpdsig = 0.25 * (pf[i, j+1, k+1] + pf[i, j, k+1] - pf[i, j+1, k-1] - pf[i, j, k-1])
        local peta = gin * (-gqj[i, j, k, 1] * dpdxi - gqj3[i, j, k] * dpdsig + (edt * (cyf[i, j, k] - vfbcn[i, k]) - sjfc[i, j, k]))
        pf[i, NJ+1, k] = pf[i, NJ, k] + peta
      end
      i += nt
    end
    @synchronize
  end
end

# faces i=0, i=NI+1: periodic
@kernel function k_mgpfill_periodic!(pf, lo)
  j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    pf[0, j, k] = pf[NI, j, k]
    pf[NI+1, j, k] = pf[1, j, k]
  end
end

function mgpfill(dtloc, pf, S=DEV)
  local edt = 0e0
  if (dtloc > 1e-16)
    edt = EPS / dtloc
  else
    println("dtime 0, set edt to 0 in mgpfill")
    edt = 0.0
  end
  local be = backend_of(S)
  if (qpr != 0)
    k_mgpfill_bottom!(be, MGPFILL_GROUPSIZE)(pf, S.gqk, S.skfc, S.czf, S.wfbcb, edt; ndrange=MGPFILL_GROUPSIZE)
    launch!(S, k_mgpfill_edges!, (1:2,), pf, S.gqk, S.skfc)
    launch!(S, k_mgpfill_top!, (1:NI, 0:NJ+1), pf)
    k_mgpfill_south!(be, MGPFILL_GROUPSIZE)(pf, S.gqj, S.gqj3, S.cyf, S.vfbcs, S.sjfc, edt; ndrange=MGPFILL_GROUPSIZE)
    k_mgpfill_north!(be, MGPFILL_GROUPSIZE)(pf, S.gqj, S.gqj3, S.cyf, S.vfbcn, S.sjfc, edt; ndrange=MGPFILL_GROUPSIZE)
  else
    launch!(S, k_mgpfill_top!, (1:NI, 0:NJ+1), pf)
  end
  launch!(S, k_mgpfill_periodic!, (0:NJ+1, 0:NK+1), pf)
end
