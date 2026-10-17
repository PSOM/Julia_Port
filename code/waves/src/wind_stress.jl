function wind_stress(udif, vdif, step)
  #  Port of waves/src/wind_stress.f90.
  #  udif(NI,NJ,NK), vdif(NI,NJ,NK): source terms from the surface wind stress (top level only).
  #  stress_top, stress_top_x, stress_top_y are the header arrays (NI,NJ).
  #  yw, ymid, yset are INTEGER in the Fortran.

  udif .= 0e0
  vdif .= 0e0

  #*********************************
  # COMPUTATION OF THE WIND STRESS

  stress_top_x .= 0e0
  stress_top_y .= 0e0
  local stressmax = 0.26e0

  local yw = 50
  local ymid = trunc(Int, 0.5 * rc_kind(NJ))
  local yset = 30

  for j in 1:div(NJ, 2)
    local yyy = rc_kind(j) - 0.5
    stress_top_y[1, j] = stressmax * 0.5 * (1e0 + tanh(((yyy - yset) / yw) * PI))
  end

  yset = NJ - 30
  for j in div(NJ, 2)+1:NJ
    local yyy = rc_kind(j) - 0.5
    stress_top_y[1, j] = stressmax * 0.5 * (1e0 + tanh((-(yyy - yset) / yw) * PI))
  end

  for j in 1:NJ
    if (abs(stress_top_y[1, j] - stressmax) < 0.001)
      stress_top_y[1, j] = stressmax
    end
    if (abs(stress_top_y[1, j]) < 0.001)
      stress_top_y[1, j] = 0e0
    end
  end

  for i in 1:NI
    for j in 1:NJ
      stress_top_y[i, j] = stress_top_y[1, j]
    end
  end

  if (step > 240)
    stress_top_y .= 0e0
  end

  #************************************
  # COMPUTATION OF THE SOURCE TERM

  local fac = 1e0 / (UL * DL * delta)
  local fact = DL / UL

  for j in 1:NJ
    for i in 1:NI
      stress_top[i, j] = sqrt(stress_top_x[i, j] * stress_top_x[i, j] + stress_top_y[i, j] * stress_top_y[i, j])
      local rhoinv = 1e0 / rho[i, j, NK]
      local Kdudzt = stress_top_x[i, j] * rhoinv * fact
      local Kdvdzt = stress_top_y[i, j] * rhoinv * fact

      udif[i, j, NK] = fac * Jac[i, j, NK] * wz[i, j, NK] * Kdudzt
      vdif[i, j, NK] = fac * Jac[i, j, NK] * wz[i, j, NK] * Kdvdzt
    end
  end

end
