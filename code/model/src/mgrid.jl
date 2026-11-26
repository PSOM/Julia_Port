# Multigrid solver for the non-hydrostatic pressure correction p (the flat vector pcorr, all
# levels one after the other; level m starts at loco[m]). V-cycles with z-line relaxation
# (linerelax) on all levels but the coarsest (sor), until max |residual| < tol.
#
# The level arrays are reshaped views into flat vectors (as in mgrid.f90): cp (19-point stencils),
# rhs and res. They are work arrays of the array set S; every entry is written (cpfine/pbc/cpcors,
# restrict, resid) before it is read, so they are not zeroed.

# views of level m: cp (19,nx,ny,nz), p (0:nx+1,0:ny+1,0:nz+1), rhs (nx,ny,nz), res (nx,ny,nz)
mg_cp(cp, L, m) = reshape(view(cp, L.loccp[m]:(L.loccp[m]-1+19*L.nx[m]*L.ny[m]*L.nz[m])), 19, L.nx[m], L.ny[m], L.nz[m])
mg_p(p, L, m) = OffsetArrays.reshape(view(p, L.loco[m]:(L.loco[m]-1+(L.nx[m]+2)*(L.ny[m]+2)*(L.nz[m]+2))),
                                     0:L.nx[m]+1, 0:L.ny[m]+1, 0:L.nz[m]+1)
mg_rhs(rhs, L, m) = reshape(view(rhs, L.loci[m]:(L.loci[m]-1+L.nx[m]*L.ny[m]*L.nz[m])), L.nx[m], L.ny[m], L.nz[m])
mg_res(res, L, m) = reshape(view(res, 1:(L.nx[m]*L.ny[m]*L.nz[m])), L.nx[m], L.ny[m], L.nz[m])

"grid sizes and offsets of the multigrid levels"
function mg_levels()
     local nx = zeros(Int, ngrid); local ny = zeros(Int, ngrid); local nz = zeros(Int, ngrid)
     local loco = zeros(Int, ngrid); local loci = zeros(Int, ngrid); local loccp = zeros(Int, ngrid)
     nx[1] = NI; ny[1] = NJ; nz[1] = NK
     loco[1] = 1; loci[1] = 1; loccp[1] = 1
     for m in 2:ngrid
          nx[m] = div(nx[m-1], 2)
          ny[m] = div(ny[m-1], 2)
          nz[m] = div(nz[m-1], 2)
          loco[m] = loco[m-1] + (nx[m-1] + 2) * (ny[m-1] + 2) * (nz[m-1] + 2)
          loci[m] = loci[m-1] + nx[m-1] * ny[m-1] * nz[m-1]
          loccp[m] = loccp[m-1] + 19 * nx[m-1] * ny[m-1] * nz[m-1]
     end
     return (; nx, ny, nz, loco, loci, loccp)
end
const MG = mg_levels()

function mgrid(p, dtime, edt, cfcdiv, S=DEV)
     local L = MG
     local cp = work(S, :mgrid_cp, rc_kind, 19 * maxint)
     local rhs = work(S, :mgrid_rhs, rc_kind, maxint)
     local res = work(S, :mgrid_res, rc_kind, int1)

     local noc = 30
     local nu1 = 2
     local nu2 = 1
     local tol_r = 1e-10

     local tol = tol_r
     local maxres = 0e0
     if (cfcdiv <= tol)
          dfill!(p, 0e0)
          return
     end
     tol = edt * tol

     cpfine(dtime, cp, rhs, S)
     for m in 2:ngrid
          # finer level m-1 is (2nx[m], 2ny[m], 2nz[m])
          local cpf = reshape(view(cp, L.loccp[m-1]:(L.loccp[m-1]-1+(19*L.nx[m]*2*L.ny[m]*2*L.nz[m]*2))), 19, L.nx[m] * 2, L.ny[m] * 2, L.nz[m] * 2)
          cpcors(L.nx[m], L.ny[m], L.nz[m], cpf, mg_cp(cp, L, m), S)
     end

     local p_base = OffsetArrays.reshape(view(p, 1:((NI+2)*(NJ+2)*(NK+2))), 0:NI+1, 0:NJ+1, 0:NK+1)
     for ncycle in 1:noc
          dfill!(S, p, 0e0, L.loco[2]:maxout)
          local m = 1
          for l in 1:nu1
               linerelax(L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), S)
               mgpfill(dtime, p_base, S)
          end
          maxres = resid(m, L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), mg_res(res, L, m), S)
          if (maxres < tol)
               return
          end
          restrict(L.nx[m], L.ny[m], L.nz[m], mg_res(res, L, m), mg_rhs(rhs, L, m + 1), S)

          for m in 2:ngrid-1
               for _ in 1:nu1
                    linerelax(L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), S)
               end
               maxres = resid(m, L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), mg_res(res, L, m), S)
               restrict(L.nx[m], L.ny[m], L.nz[m], mg_res(res, L, m), mg_rhs(rhs, L, m + 1), S)
          end

          for m in ngrid:-1:2
               if m == ngrid
                    sor(L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), S)
               else
                    for _ in 1:nu2
                         linerelax(L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), S)
                    end
               end
               efill(L.nx[m], L.ny[m], L.nz[m], mg_p(p, L, m), S)
               # the finer level m-1 has (2nx[m], 2ny[m], 2nz[m]) interior points
               local p_fin = OffsetArrays.reshape(view(p, L.loco[m-1]:(L.loco[m-1]-1+((2*L.nx[m]+2)*(2*L.ny[m]+2)*(2*L.nz[m]+2)))),
                                                  0:2*L.nx[m]+1, 0:2*L.ny[m]+1, 0:2*L.nz[m]+1)
               prolong(L.nx[m], L.ny[m], L.nz[m], mg_p(p, L, m), p_fin, S)
          end
     end

     # not converged after noc cycles: two more relaxations on the finest level
     local m = 1
     for l in 1:nu1
          linerelax(L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), S)
          mgpfill(dtime, p_base, S)
     end
     maxres = resid(m, L.nx[m], L.ny[m], L.nz[m], mg_cp(cp, L, m), mg_p(p, L, m), mg_rhs(rhs, L, m), mg_res(res, L, m), S)
     maxres = maxres / edt
end
