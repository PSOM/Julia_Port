@kernel function k_evalrho_sT!(rhonew, s, T, nl2, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds rhonew[i, j, k] = potdens(s[i, j, k, nl2], T[i, j, k, nl2])
end

function evalrho_sT(rhonew, nl2, S=DEV)
  launch!(S, k_evalrho_sT!, (0:NI+1, 0:NJ+1, 0:NK+1), rhonew, S.s, S.T, nl2)
end
