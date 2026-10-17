function sponge(n, dum_time)
  #  Port of waves/src/sponge.f90: sponge layers at the southern/northern boundaries (below splz)
  #  and a relaxation of the upper splz levels.
  local spl = 15
  local splz = 60
  local gamma_T = 1e0 / (dum_time)

  for j in 0:spl+1
    # NOTE: (j-1)/spl is an INTEGER division in sponge.f90 (j and spl are integers), so the
    # cosine taper is a step: dummy = gamma_T for j=0..15 and cos(pi/2)*gamma_T at j=16.
    # Kept identical to the Fortran; a smooth taper was probably intended.
    local dummy = gamma_T * cos(div(j - 1, spl) * PI / 2)
    if (j == 0)
      dummy = gamma_T
    end

    for k in splz+1:NK+1
      for i in 0:NI+1
        s[i, j, k, n] = s[i, j, k, n] - dum_time * dummy * (s[i, j, k, n] - rho_refS[k])
        u[i, j, k, n] = u[i, j, k, n] - dum_time * dummy * u[i, j, k, n]
        v[i, j, k, n] = v[i, j, k, n] - dum_time * dummy * v[i, j, k, n]
        w[i, j, k, n] = w[i, j, k, n] - dum_time * dummy * w[i, j, k, n]
      end
    end

  end

  for j in NJ-spl:NJ+1

    # integer division as in the Fortran (see note above)
    local dummy = gamma_T * sin(div(j - (NJ - spl), spl) * PI / 2)
    if (j == NJ + 1)
      dummy = gamma_T
    end
    for k in splz+1:NK+1
      for i in 0:NI+1
        s[i, j, k, n] = s[i, j, k, n] - dum_time * dummy * (s[i, j, k, n] - rho_refN[k])
        u[i, j, k, n] = u[i, j, k, n] - dum_time * dummy * u[i, j, k, n]
        v[i, j, k, n] = v[i, j, k, n] - dum_time * dummy * v[i, j, k, n]
        w[i, j, k, n] = w[i, j, k, n] - dum_time * dummy * w[i, j, k, n]
      end
    end

  end

  local rinn = 2e0 * PI / (ffc[div(NI, 2), div(NJ, 2)] * FPAR)
  local rout = rinn / 1e3

  for k in 0:splz

    local rdum = (rc_kind(splz - k) * rout + rc_kind(k) * rinn) / rc_kind(splz)
    local rrr = 0e0
    if (rdum != 0)
      rrr = 1e0 / rdum
    end

    for j in 0:NJ+1
      for i in 0:NI+1

        local s_relax = (rc_kind(splz - k) * rho_refB[j] + rc_kind(k) * s[i, j, k, n]) / rc_kind(splz)
        local u_relax = (rc_kind(k) * u[i, j, k, n]) / rc_kind(splz)
        local v_relax = (rc_kind(k) * v[i, j, k, n]) / rc_kind(splz)

        s[i, j, k, n] = s[i, j, k, n] - dum_time * rrr * (s[i, j, k, n] - s_relax)
        u[i, j, k, n] = u[i, j, k, n] - dum_time * rrr * (u[i, j, k, n] - u_relax)
        v[i, j, k, n] = v[i, j, k, n] - dum_time * rrr * (v[i, j, k, n] - v_relax)

      end
    end

  end

end
