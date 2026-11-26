# mprove: iterative improvement of the free-surface solution h of ch*h = oldrhs (see hsolve):
# solve ch*he = ch*h - oldrhs for the error he by multi-colour SOR, then h -= he.

@kernel function k_mprove_rhs!(rhs, h, ch, oldrhs, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds begin
    local im1 = i == 1 ? NI : i - 1
    local ip1 = i == NI ? 1 : i + 1
    rhs[i, j] = (ch[1, i, j] * h[i, j]
               + ch[2, i, j] * h[ip1, j]
               + ch[3, i, j] * h[im1, j]
               + ch[4, i, j] * h[i, j+1]
               + ch[5, i, j] * h[i, j-1]
               + ch[6, i, j] * h[ip1, j+1]
               + ch[7, i, j] * h[im1, j+1]
               + ch[8, i, j] * h[ip1, j-1]
               + ch[9, i, j] * h[im1, j-1]) - oldrhs[i, j]
  end
end

# boundary condition on the error: since the old bc dh/dn is done perfectly, d(he)/dn = 0
@kernel function k_mprove_bcy!(he, lo)
  i, = @index(Global, NTuple) .+ lo
  @inbounds begin
    he[i, 0] = he[i, 1]
    he[i, NJ+1] = he[i, NJ]
  end
end
# periodic in x (after k_mprove_bcy!: also copies the corners)
@kernel function k_mprove_bcx!(he, lo)
  j, = @index(Global, NTuple) .+ lo
  @inbounds begin
    he[0, j] = he[NI, j]
    he[NI+1, j] = he[1, j]
  end
end

@kernel function k_mprove_sub!(h, he, lo)
  i, j = @index(Global, NTuple) .+ lo
  @inbounds h[i, j] -= he[i, j]
end

function mprove(h, ch, oldrhs, dtime, S=DEV)
  local he = work(S, :mprove_he, rc_kind, 0:NI+1, 0:NJ+1)
  local rhs = work(S, :mprove_rhs, rc_kind, NI, NJ)
  local tol = 1e-10
  dfill!(he, 0e0)

  # new RHS: A*h - b
  launch!(S, k_mprove_rhs!, (1:NI, 1:NJ), rhs, h, ch, oldrhs)

  local rlx = 1.72e0
  for iter in 1:1000
    sor9_sweep!(he, ch, rhs, rlx, S)
    launch!(S, k_mprove_bcy!, (1:NI,), he)
    launch!(S, k_mprove_bcx!, (0:NJ+1,), he)

    local maxres = res9_max(he, ch, rhs, S)
    if (maxres > 3000)
      println("mprove.jl: STOP. res too large, maxres=", maxres)
      exit(1)
    end
    if (maxres < tol)
      break
    end
  end
  launch!(S, k_mprove_sub!, (0:NI+1, 0:NJ+1), h, he)
end
