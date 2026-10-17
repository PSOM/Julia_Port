# vface: face fluxes uf, vf, wf corrected with the non-hydrostatic pressure pf (the first-level part
# of the flat multigrid vector).
@kernel function k_vface_u!(uf, pf, gqi, gqi3, cxf, sifc, dte, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local px = ((pf[i+1, j, k] - pf[i, j, k]) * gqi[i, j, k, 1]
    + 0.25e0 * (pf[i+1, j+1, k] + pf[i, j+1, k] - pf[i+1, j-1, k] - pf[i, j-1, k]) * gqi[i, j, k, 2]
     + 0.25e0 * (pf[i+1, j, k+1] + pf[i, j, k+1] - pf[i+1, j, k-1] - pf[i, j, k-1]) * gqi3[i, j, k])
    uf[i, j, k] = cxf[i, j, k] - dte * (px + sifc[i, j, k])
  end
end

# periodic: uf(0) = uf(NI) (after k_vface_u!)
@kernel function k_vface_u0!(uf, lo)
  j, k = @index(Global, NTuple) .+ lo
  @inbounds uf[0, j, k] = uf[NI, j, k]
end

@kernel function k_vface_v!(vf, pf, gqj, gqj3, cyf, sjfc, vfbcs, vfbcn, dte, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds if (j == 0)
    vf[i, j, k] = vfbcs[i, k]
  elseif (j == NJ)
    vf[i, j, k] = vfbcn[i, k]
  else
    local py = ((pf[i, j+1, k] - pf[i, j, k]) * gqj[i, j, k, 2]
      + 0.25e0 * (pf[i+1, j+1, k] + pf[i+1, j, k] - pf[i-1, j+1, k] - pf[i-1, j, k]) * gqj[i, j, k, 1]
      + 0.25e0 * (pf[i, j+1, k+1] + pf[i, j, k+1] - pf[i, j+1, k-1] - pf[i, j, k-1]) * gqj3[i, j, k])
    vf[i, j, k] = cyf[i, j, k] - dte * (py + sjfc[i, j, k])
  end
end

@kernel function k_vface_w!(wf, pf, gqk, czf, skfc, wfbcb, dte, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds if k == 0
    wf[i, j, 0] = wfbcb[i, j]
  else
    local pz = ((pf[i, j, k+1] - pf[i, j, k]) * gqk[i, j, k, 3]
      + 0.25e0 * (pf[i+1, j, k+1] + pf[i+1, j, k] - pf[i-1, j, k+1] - pf[i-1, j, k]) * gqk[i, j, k, 1]
      + 0.25e0 * (pf[i, j+1, k+1] + pf[i, j+1, k] - pf[i, j-1, k+1] - pf[i, j-1, k]) * gqk[i, j, k, 2])
    wf[i, j, k] = czf[i, j, k] - dte * (pz + skfc[i, j, k])
  end
end

function vface(pf, dtimel::rc_kind, S=DEV)
  local pf3 = OffsetArrays.reshape(view(pf, 1:((NI+2)*(NJ+2)*(NK+2))), 0:NI+1, 0:NJ+1, 0:NK+1)
  local dte = dtimel / EPS
  launch!(S, k_vface_u!, (1:NI, 1:NJ, 1:NK), S.uf, pf3, S.gqi, S.gqi3, S.cxf, S.sifc, dte)
  launch!(S, k_vface_u0!, (1:NJ, 1:NK), S.uf)
  launch!(S, k_vface_v!, (1:NI, 0:NJ, 1:NK), S.vf, pf3, S.gqj, S.gqj3, S.cyf, S.sjfc, S.vfbcs, S.vfbcn, dte)
  launch!(S, k_vface_w!, (1:NI, 1:NJ, 0:NK), S.wf, pf3, S.gqk, S.czf, S.skfc, S.wfbcb, dte)
end
