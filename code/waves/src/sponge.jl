#  Port of waves/src/sponge.f90: sponge layers at the southern/northern boundaries (below splz)
#  and a relaxation of the upper splz levels.

# southern sponge, j = 0:spl+1, k = splz+1:NK+1
@kernel function k_sponge_south!(s, u, v, w, rho_refS, n, dum_time, gamma_T, spl, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    # smooth cosine taper (the Fortran originally used an integer division here, giving a step)
    local dummy = gamma_T * cos((j - 1) / spl * PI / 2)
    if (j == 0)
      dummy = gamma_T
    end
    s[i, j, k, n] = s[i, j, k, n] - dum_time * dummy * (s[i, j, k, n] - rho_refS[k])
    u[i, j, k, n] = u[i, j, k, n] - dum_time * dummy * u[i, j, k, n]
    v[i, j, k, n] = v[i, j, k, n] - dum_time * dummy * v[i, j, k, n]
    w[i, j, k, n] = w[i, j, k, n] - dum_time * dummy * w[i, j, k, n]
  end
end

# northern sponge, j = NJ-spl:NJ+1, k = splz+1:NK+1
@kernel function k_sponge_north!(s, u, v, w, rho_refN, n, dum_time, gamma_T, spl, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local dummy = gamma_T * sin((j - (NJ - spl)) / spl * PI / 2)
    if (j == NJ + 1)
      dummy = gamma_T
    end
    s[i, j, k, n] = s[i, j, k, n] - dum_time * dummy * (s[i, j, k, n] - rho_refN[k])
    u[i, j, k, n] = u[i, j, k, n] - dum_time * dummy * u[i, j, k, n]
    v[i, j, k, n] = v[i, j, k, n] - dum_time * dummy * v[i, j, k, n]
    w[i, j, k, n] = w[i, j, k, n] - dum_time * dummy * w[i, j, k, n]
  end
end

# relaxation of the levels k = 0:splz (time scale from the inertial period at the domain centre)
@kernel function k_sponge_top!(s, u, v, ffc, rho_refB, n, dum_time, splz, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds begin
    local rinn = 2e0 * PI / (ffc[div(NI, 2), div(NJ, 2)] * FPAR)
    local rout = rinn / 1e3
    local rdum = (rc_kind(splz - k) * rout + rc_kind(k) * rinn) / rc_kind(splz)
    local rrr = 0e0
    if (rdum != 0)
      rrr = 1e0 / rdum
    end

    local s_relax = (rc_kind(splz - k) * rho_refB[j] + rc_kind(k) * s[i, j, k, n]) / rc_kind(splz)
    local u_relax = (rc_kind(k) * u[i, j, k, n]) / rc_kind(splz)
    local v_relax = (rc_kind(k) * v[i, j, k, n]) / rc_kind(splz)

    s[i, j, k, n] = s[i, j, k, n] - dum_time * rrr * (s[i, j, k, n] - s_relax)
    u[i, j, k, n] = u[i, j, k, n] - dum_time * rrr * (u[i, j, k, n] - u_relax)
    v[i, j, k, n] = v[i, j, k, n] - dum_time * rrr * (v[i, j, k, n] - v_relax)
  end
end

function sponge(n, dum_time, S=DEV)
  local spl = 15
  local splz = 60
  local gamma_T = 1e0 / (dum_time)

  # the three regions are applied one after the other, as in the Fortran (they would only interact
  # if they overlapped)
  launch!(S, k_sponge_south!, (0:NI+1, 0:spl+1, splz+1:NK+1), S.s, S.u, S.v, S.w, S.rho_refS, n, dum_time, gamma_T, spl)
  launch!(S, k_sponge_north!, (0:NI+1, NJ-spl:NJ+1, splz+1:NK+1), S.s, S.u, S.v, S.w, S.rho_refN, n, dum_time, gamma_T, spl)
  launch!(S, k_sponge_top!, (0:NI+1, 0:NJ+1, 0:splz), S.s, S.u, S.v, S.ffc, S.rho_refB, n, dum_time, splz)
end
