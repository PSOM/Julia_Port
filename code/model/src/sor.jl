# sor: point SOR on the coarsest multigrid level (8 sweeps, lexicographic order k, j, i as in
# sor.f90). To run it in parallel without changing the result, each sweep is done as a wavefront:
# with the 19-point stencil, every neighbour that the lexicographic order has already updated has a
# smaller t = i + 2j + 3k, and every neighbour it has not yet updated a larger t. So all points of one
# hyperplane t = const are independent, and sweeping the hyperplanes in increasing t reproduces the
# lexicographic result exactly. One workgroup does all hyperplanes, with a barrier between them.
const SOR_GROUPSIZE = 256

@kernel function k_sor_wavefront!(p, cp, fn, nxm, nym, nzm, rlx, niter)
      tid = @index(Local, Linear)
      nt = @uniform @groupsize()[1]
      for iter in 1:niter
            for t in 6:(nxm + 2 * nym + 3 * nzm)
                  # points with i + 2j + 3k = t; thread tid takes every nt-th one, enumerated over (k, j)
                  local n = tid
                  for k in 1:nzm, j in 1:nym
                        local i = t - 2 * j - 3 * k
                        if 1 <= i <= nxm
                              if n == 1
                                    @inbounds begin
                                          local pstar = ( cp[2,i,j,k]*p[i+1,j,k]
                                                        +cp[3,i,j,k ]*p[i,j+1,k]
                                                        +cp[4,i,j,k ]*p[i-1,j,k]
                                                        +cp[5,i,j,k ]*p[i,j-1,k]
                                                        +cp[6,i,j,k ]*p[i,j,k+1]
                                                        +cp[7,i,j,k ]*p[i,j,k-1]
                                                        +cp[8,i,j,k ]*p[i-1,j+1,k]
                                                        +cp[9,i,j,k ]*p[i-1,j-1,k]
                                                        +cp[10,i,j,k]*p[i+1,j-1,k]
                                                        +cp[11,i,j,k]*p[i+1,j+1,k]
                                                        +cp[12,i,j,k]*p[i-1,j,k-1]
                                                        +cp[13,i,j,k]*p[i+1,j,k-1]
                                                        +cp[14,i,j,k]*p[i+1,j,k+1]
                                                        +cp[15,i,j,k]*p[i-1,j,k+1]
                                                        +cp[16,i,j,k]*p[i,j-1,k-1]
                                                        +cp[17,i,j,k]*p[i,j+1,k-1]
                                                        +cp[18,i,j,k]*p[i,j+1,k+1]
                                                        +cp[19,i,j,k]*p[i,j-1,k+1]
                                                        - fn[i,j,k] )/(-cp[1,i,j,k])
                                          p[i,j,k] = (1e0 -rlx)*p[i,j,k] +rlx*pstar
                                    end
                                    n = nt
                              else
                                    n -= 1
                              end
                        end
                  end
                  @synchronize
            end
      end
end

function sor(nxm, nym, nzm, cp, p, fn, S=DEV)
      local rlx = 1.8e0
      k_sor_wavefront!(backend_of(S), SOR_GROUPSIZE)(p, cp, fn, nxm, nym, nzm, rlx, 8; ndrange=SOR_GROUPSIZE)
end
