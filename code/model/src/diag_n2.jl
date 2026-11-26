function diag_n2()
  local DLinv = 1e0 / DL
  # x-averages of rho over the full declared row 0:NI+1, divided by NI (as SUM(rho(:,j,k))/NI in diag_n2.f90).
  # Computed once per (j,k) instead of inside the i loop; sequential summation like the Fortran SUM.
  local rhomean = OffsetArray(zeros(rc_kind, NJ + 2, NK + 2), 0:NJ+1, 0:NK+1)
  for k in 0:NK+1
    for j in 0:NJ+1
      local acc = 0e0
      for i in 0:NI+1
        acc += rho[i, j, k]
      end
      rhomean[j, k] = acc / NI
    end
  end
  for k in 1:NK
    for j in 1:NJ
      for i in 1:NI

        local rdy = 0.5e0 * ((rho[i, j+1, k] - rhomean[j+1, k]) - (rho[i, j-1, k] - rhomean[j-1, k])) * vy[i, j] / LEN
        local rdz = 0e0
        local rpdz = 0e0
        if (k == NK)
          rdz = (rho[i, j, k] - rho[i, j, k-1]) * wz[i, j, k] * DLinv
          rpdz = (rho[i, j, k] - rhomean[j, k] - (rho[i, j, k-1] - rhomean[j, k-1])) * wz[i, j, k] * DLinv
        elseif (k == 1)
          rdz = (rho[i, j, k+1] - rho[i, j, k]) * wz[i, j, k] * DLinv
          rpdz = (rho[i, j, k+1] - rhomean[j, k+1] - (rho[i, j, k] - rhomean[j, k])) * wz[i, j, k] * DLinv
        else
          rdz = 0.5e0 * (rho[i, j, k+1] - rho[i, j, k-1]) * wz[i, j, k] * DLinv
          rpdz = 0.5e0 * (rho[i, j, k+1] - rhomean[j, k+1] - (rho[i, j, k-1] - rhomean[j, k-1])) * wz[i, j, k] * DLinv
        end

        freqby[i, j, k] = (-gpr * 10e0 / R0) * rdy
        freqbz[i, j, k] = (-gpr * 10e0 / R0) * rpdz
        freqN2[i, j, k] = (-gpr * 10e0 / R0) * rdz

      end
    end
  end
end
