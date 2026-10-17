@kernel function k_facediv!(dv, uf, vf, wf, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local ufdx = (uf[i, j, k] - uf[i-1, j, k])
    local vfdy = (vf[i, j, k] - vf[i, j-1, k])
    local wfdz = (wf[i, j, k] - wf[i, j, k-1])
    dv[i, j, k] = abs(ufdx + vfdy + wfdz)
  end
end

"max |divergence| of the face fluxes"
function facediv(dtimel, S=DEV)
  local dv = work(S, :div3, rc_kind, NI, NJ, NK)
  launch!(S, k_facediv!, (1:NI, 1:NJ, 1:NK), dv, S.uf, S.vf, S.wf)
  return maximum(dv)
end
