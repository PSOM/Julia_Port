# Face velocities cxf, cyf, czf: hydrostatic (free-surface and face-source) update.

@kernel function k_vhydro_cxf!(cxf, hxn, h, gi, sifc, dte, kappah, kaph1, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local hxi = h[i+1, j] - h[i, j]
    local heta = 0.25e0 * ( h[i+1, j+1] + h[i, j+1] - h[i+1, j-1] - h[i, j-1])
    local hx = gi[i, j, k, 1] * hxi + gi[i, j, k, 2] * heta
    local gradh = gpr * (kappah * hx + kaph1 * hxn[i, j, k])
    cxf[i, j, k] -= dte * (gradh + sifc[i, j, k])
    hxn[i, j, k] = hx
  end
end

# periodic in i
@kernel function k_vhydro_cxf_bc!(cxf, hxn, lo)
  j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    cxf[0, j, k] = cxf[NI, j, k]
    hxn[0, j, k] = hxn[NI, j, k]
  end
end

@kernel function k_vhydro_cyf!(cyf, hyn, h, gj, sjfc, dte, kappah, kaph1, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local hxi = 0.25e0 * (h[i+1, j+1] + h[i+1, j] - h[i-1, j+1] - h[i-1, j])
    local heta = h[i, j+1] - h[i, j]
    local hy = gj[i, j, k, 1] * hxi + gj[i, j, k, 2] * heta
    local gradh = gpr * (kappah * hy + kaph1 * hyn[i, j, k])
    cyf[i, j, k] -= dte * (gradh + sjfc[i, j, k])
    hyn[i, j, k] = hy
  end
end

@kernel function k_vhydro_cyf_bc!(cyf, vfbcn, vfbcs, lo)
  i, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    cyf[i, NJ, k] = vfbcn[i,k]
    cyf[i, 0, k] = vfbcs[i,k]
  end
end

@kernel function k_vhydro_czf!(czf, p, gqk, skfc, dte, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local pz = (p[i, j, k+1] - p[i, j, k]) * gqk[i, j, k, 3] + 0.25e0 * (p[i+1, j, k+1] + p[i+1, j, k] - p[i-1, j, k+1] - p[i-1, j, k]) * gqk[i, j, k, 1] + 0.25e0 * (p[i, j+1, k+1] + p[i, j+1, k] - p[i, j-1, k+1] - p[i, j-1, k]) * gqk[i, j, k, 2]
    czf[i, j, k] -= dte * (pz + skfc[i, j, k])
  end
end

function vhydro(dtimel, S=DEV)
  local dte = dtimel / EPS
  local kaph1 = 1e0 - kappah

  launch!(S, k_vhydro_cxf!, (1:NI, 1:NJ, 1:NK), S.cxf, S.hxn, S.h, S.gi, S.sifc, dte, kappah, kaph1)
  launch!(S, k_vhydro_cxf_bc!, (1:NJ, 1:NK), S.cxf, S.hxn)

  launch!(S, k_vhydro_cyf!, (1:NI, 0:NJ, 1:NK), S.cyf, S.hyn, S.h, S.gj, S.sjfc, dte, kappah, kaph1)
  launch!(S, k_vhydro_cyf_bc!, (1:NI, 1:NK), S.cyf, S.vfbcn, S.vfbcs)

  launch!(S, k_vhydro_czf!, (1:NI, 1:NJ, 0:NK), S.czf, S.p, S.gqk, S.skfc, dte)

  uvchy(dtimel, S)
end
