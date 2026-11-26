# efill: ghost points of a coarse multigrid level (zero-gradient, p=-p above the top). The
# launches follow the order of the original loops, because later ones read ghost points
# written by earlier ones.
@kernel function k_efill_y!(p, nym, lo)                 # j ghosts, k=1..nzm
   i, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      p[i, 0, k] = p[i, 1, k]
      p[i, nym+1, k] = p[i, nym, k]
   end
end
@kernel function k_efill_x!(p, nxm, lo)                 # i ghosts, k=1..nzm
   j, k = @index(Global, NTuple) .+ lo
   @inbounds begin
      p[0, j, k] = p[1, j, k]
      p[nxm+1, j, k] = p[nxm, j, k]
   end
end
@kernel function k_efill_bottom!(p, lo)                 # k=0, interior i, j
   i, j = @index(Global, NTuple) .+ lo
   @inbounds p[i, j, 0] = p[i, j, 1]
end
@kernel function k_efill_top!(p, nzm, lo)               # k=nzm+1, all i, j
   i, j = @index(Global, NTuple) .+ lo
   @inbounds p[i, j, nzm+1] = -p[i, j, nzm]
end
@kernel function k_efill_corners!(p, nxm, nym, lo)      # vertical edges, k=1..nzm
   k, = @index(Global, NTuple) .+ lo
   @inbounds begin
      p[0, 0, k] = p[1, 1, k]
      p[nxm+1, 0, k] = p[nxm, 1, k]
      p[0, nym+1, k] = p[1, nym, k]
      p[nxm+1, nym+1, k] = p[nxm, nym, k]
   end
end
@kernel function k_efill_bottom_y!(p, nym, lo)          # k=0, j ghosts
   i, = @index(Global, NTuple) .+ lo
   @inbounds begin
      p[i, 0, 0] = p[i, 1, 1]
      p[i, nym+1, 0] = p[i, nym, 1]
   end
end
@kernel function k_efill_bottom_x!(p, nxm, lo)          # k=0, i ghosts
   j, = @index(Global, NTuple) .+ lo
   @inbounds begin
      p[0, j, 0] = p[1, j, 1]
      p[nxm+1, j, 0] = p[nxm, j, 1]
   end
end
@kernel function k_efill_bottom_corners!(p, nxm, nym, lo)
   @inbounds begin
      p[0, 0, 0] = p[1, 1, 1]
      p[0, nym+1, 0] = p[1, nym, 1]
      p[nxm+1, 0, 0] = p[nxm, 1, 1]
      p[nxm+1, nym+1, 0] = p[nxm, nym, 1]
   end
end

function efill(nxm, nym, nzm, p, S=DEV)
   launch!(S, k_efill_y!, (1:nxm, 1:nzm), p, nym)
   launch!(S, k_efill_x!, (1:nym, 1:nzm), p, nxm)
   launch!(S, k_efill_bottom!, (1:nxm, 1:nym), p)
   launch!(S, k_efill_top!, (0:nxm+1, 0:nym+1), p, nzm)
   launch!(S, k_efill_corners!, (1:nzm,), p, nxm, nym)
   launch!(S, k_efill_bottom_y!, (1:nxm,), p, nym)
   launch!(S, k_efill_bottom_x!, (1:nym,), p, nxm)
   launch!(S, k_efill_bottom_corners!, (1:1,), p, nxm, nym)
end
