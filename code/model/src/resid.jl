# resid: residual res = fn - A*p of the pressure equation on one multigrid level; returns max |res|
@kernel function k_resid!(res, cp, p, fn, lo)
      i, j, k = @index(Global, NTuple) .+ lo
      @inbounds res[i, j, k] = fn[i, j, k] -
                               (cp[1, i, j, k] * p[i, j, k]
                                + cp[2, i, j, k] * p[i+1, j, k]
                                + cp[3, i, j, k] * p[i, j+1, k]
                                + cp[4, i, j, k] * p[i-1, j, k]
                                + cp[5, i, j, k] * p[i, j-1, k]
                                + cp[6, i, j, k] * p[i, j, k+1]
                                + cp[7, i, j, k] * p[i, j, k-1]
                                + cp[8, i, j, k] * p[i-1, j+1, k]
                                + cp[9, i, j, k] * p[i-1, j-1, k]
                                + cp[10, i, j, k] * p[i+1, j-1, k]
                                + cp[11, i, j, k] * p[i+1, j+1, k]
                                + cp[12, i, j, k] * p[i-1, j, k-1]
                                + cp[13, i, j, k] * p[i+1, j, k-1]
                                + cp[14, i, j, k] * p[i+1, j, k+1]
                                + cp[15, i, j, k] * p[i-1, j, k+1]
                                + cp[16, i, j, k] * p[i, j-1, k-1]
                                + cp[17, i, j, k] * p[i, j+1, k-1]
                                + cp[18, i, j, k] * p[i, j+1, k+1]
                                + cp[19, i, j, k] * p[i, j-1, k+1])
end

function resid(m, nxm, nym, nzm, cp, p, fn, res, S=DEV)
      launch!(S, k_resid!, (1:nxm, 1:nym, 1:nzm), res, cp, p, fn)
      local maxres = maximum(abs, res)

      if (maxres > 3000e0)
            println("STOP. res too large, i,j,k,maxres=",maxres)
            exit(1)
      end

      return maxres
end
