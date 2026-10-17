@kernel function k_diag_energy!(e, Jac, u, v, w, cst, lo)
  i, j, k = @index(Global, NTuple) .+ lo
  @inbounds e[i, j, k] = Jac[i, j, k] * (u[i, j, k, 0]^2 + v[i, j, k, 0]^2 + cst * w[i, j, k, 0]^2)
end

"print the total kinetic energy (sum over the interior of Jac*(u^2 + v^2 + (EPS*delta)^2 w^2))"
function diag_energy(step, S=DEV)
  local cst = EPS * EPS * delta * delta
  local e = work(S, :diag_energy_e, rc_kind, NI, NJ, NK)
  launch!(S, k_diag_energy!, (1:NI, 1:NJ, 1:NK), e, S.Jac, S.u, S.v, S.w, cst)
  local enkin = sum(e)
  println("#total kinetic energy = ", rpad(step, 10, " "), " ", enkin)
end
