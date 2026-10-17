# Convective adjustment (port of conadjust_sT.f90): where the density is unstable between k and k+1,
# mix s and T of the two cells (thickness-weighted) and set conv = 1.

# one work item per column (i, j): the k loop goes down and each level uses the rho, s, T updated
# at the level above
@kernel function k_conadjust!(conv, rho, zf, s, T, n, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds for k in NK-1:-1:1
    conv[i, j, k] = 0
    local rhtop = rho[i, j, k+1]
    local dzupper = zf[i, j, k+1] - zf[i, j, k]

    local rh = rho[i, j, k]
    local dz = zf[i, j, k] - zf[i, j, k-1]

    if (rh < rhtop)
      conv[i, j, k] = 1
      local zinv = 1e0 / (dzupper + dz)
      s[i, j, k, n] = (dzupper * s[i, j, k+1, n] + dz * s[i, j, k, n]) * zinv
      s[i, j, k+1, n] = s[i, j, k, n]
      T[i, j, k, n] = (dzupper * T[i, j, k+1, n] + dz * T[i, j, k, n]) * zinv
      T[i, j, k+1, n] = T[i, j, k, n]
      rho[i, j, k] = potdens(s[i, j, k, n], T[i, j, k, n])
      rho[i, j, k+1] = rho[i, j, k]
    end
  end
end

# The tracer part, as in the Julia port this replaces: Tr(k) is "mixed" with itself and copied to
# (i, j+1, k) (not (i, j, k+1)). Because of the j+1 write, the columns depend on each other along j:
# one work item per (i, k), with the j loop in the original (increasing) order.
@kernel function k_conadjust_tr!(Tr, conv, zf, n, lo)
  i, k = @index(Global, NTuple) .+ lo
  @inbounds for j in 1:NJ
    if (conv[i, j, k] == 1)
      local dzupper = zf[i, j, k+1] - zf[i, j, k]
      local dz = zf[i, j, k] - zf[i, j, k-1]
      local zinv = 1e0 / (dzupper + dz)
      for it in 1:ntr
        Tr[it, i, j, k, n] = (dzupper * Tr[it, i, j, k, n] + dz * Tr[it, i, j, k, n]) * zinv
        Tr[it, i, j+1, k, n] = Tr[it, i, j, k, n]
      end
    end
  end
end

function conadjust(stepl, n, S=DEV)
  launch!(S, k_conadjust!, (1:NI, 1:NJ), S.conv, S.rho, S.zf, S.s, S.T, n)
  launch!(S, k_conadjust_tr!, (1:NI, 1:NK-1), S.Tr, S.conv, S.zf, n)

  if (mod(stepl - 1, 100) == 0)
    dfill!(S.con100, 0)
  else
    daddto!(S, S.con100, S.conv, 0:NI+1, 0:NJ+1, 0:NK+1)
  end
end
