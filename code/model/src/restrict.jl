# restrict: fine-level residual fin (nxm x nym x nzm) -> coarse-level rhs cor
@kernel function k_restrict!(cor, fin, coeff, lo)
      i, j, k = @index(Global, NTuple) .+ lo
      @inbounds begin
            local ii = 2 * i - 1
            local jj = 2 * j - 1
            local kk = 2 * k - 1
            cor[i,j,k] = coeff*(fin[ii,jj,kk]+fin[ii,jj+1,kk]
                          +fin[ii+1,jj,kk]+fin[ii+1,jj+1,kk] +
                          fin[ii,jj,kk+1] + fin[ii,jj+1,kk+1]
                          + fin[ii+1,jj,kk+1] + fin[ii+1,jj+1,kk+1] )
      end
end

function restrict(nxm, nym, nzm, fin, cor, S=DEV)
      local coeff = 0.125 * 4e0
      launch!(S, k_restrict!, (1:div(nxm, 2), 1:div(nym, 2), 1:div(nzm, 2)), cor, fin, coeff)
end
