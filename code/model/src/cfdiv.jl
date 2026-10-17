@kernel function k_cfdiv!(dv, cxf, cyf, czf, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local cxfdx = (cxf[i, j, k] - cxf[i-1, j, k])
    local cyfdy = (cyf[i, j, k] - cyf[i, j-1, k])
    local czfdz = (czf[i, j, k] - czf[i, j, k-1])
    dv[i, j, k] = abs(cxfdx + cyfdy + czfdz)
  end
end

"max |divergence| of the predicted face fluxes cxf, cyf, czf"
function cfdiv(maxdiv::Ref{rc_kind}, S=DEV)
  local dv = work(S, :div3, rc_kind, NI, NJ, NK)
  launch!(S, k_cfdiv!, (1:NI, 1:NJ, 1:NK), dv, S.cxf, S.cyf, S.czf)
  maxdiv[] = maximum(dv)
end
