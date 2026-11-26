@kernel function k_evalrho_rho!(rhonew, s, n, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds rhonew[i, j, k] = s[i, j, k, n]
end

function evalrho_rho(rhonew, n, S=DEV)
  launch!(S, k_evalrho_rho!, (0:NI+1, 0:NJ+1, 0:NK+1), rhonew, S.s, n)
end
