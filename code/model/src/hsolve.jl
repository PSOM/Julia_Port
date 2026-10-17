
# precalculates hfill sums to avoid recalculations
# (sequential k=1..NK summation, same order as the Fortran hfill loops)
seqsum(v) = foldl(+, v; init=0e0)
function hsolve_getSum()
    return (
      sumvf = [seqsum(@view vfbcs[i, 1:NK]) for i in 1:NI],
      sumcyfj = [seqsum(@view cyf[i, 0, 1:NK]) for i in 1:NI],
      sumsjfj = [seqsum(@view sjfc[i, 0, 1:NK]) for i in 1:NI],
      sumhynj = [seqsum(@view hyn[i, 0, 1:NK]) for i in 1:NI],
      sumgjj = [(seqsum(@view gj[i, 0, 1:NK, 1]), seqsum(@view gj[i, 0, 1:NK, 2])) for i in 1:NI],

      sumvfn = [seqsum(@view vfbcn[i, 1:NK]) for i in 1:NI],
      sumcyfnj = [seqsum(@view cyf[i, NJ, 1:NK]) for i in 1:NI],
      sumsjfnj = [seqsum(@view sjfc[i, NJ, 1:NK]) for i in 1:NI],
      sumhynnj = [seqsum(@view hyn[i, NJ, 1:NK]) for i in 1:NI],
      sumgjnj = [(seqsum(@view gj[i, NJ, 1:NK, 1]), seqsum(@view gj[i, NJ, 1:NK, 2])) for i in 1:NI],
    )
end


function hsolve(h,oldh,hdt,dtime)
  local ch = zeros(rc_kind, (9, NI, NJ))
  local rhs = zeros(rc_kind, (NI, NJ))

  local tol::rc_kind = 1e-12
  local dti::rc_kind = 1e0 / dtime
  local kount::Int = 0

  local sums = hsolve_getSum()
  chfine(dtime, ch, rhs)
  #@ccall "./PSOM_LIB.so".chfine_(Ref(dtime)::Ref{rc_kind}, pointer(ch)::Ptr{rc_kind}, pointer(rhs)::Ptr{rc_kind})::Cvoid


  local rlx::rc_kind = 1.72e0
  local nci = ncolours_i(NI)

  for iter in 1:100
    # multi-colour SOR sweep (see colour_irange in linerelax_periodicew.jl), as in the Fortran
    for cj in 1:2, ci in 1:nci
      for j in cj:2:NJ, i in colour_irange(NI, ci, nci)
        local im1 = i == 1 ? NI : i - 1
        local ip1 = i == NI ? 1 : i + 1
        local hstar = ( ch[2, i, j] * h[ip1, j]
                      + ch[3, i, j] * h[im1, j]
                      + ch[4, i, j] * h[i, j+1]
                      + ch[5, i, j] * h[i, j-1]
                      + ch[6, i, j] * h[ip1, j+1]
                      + ch[7, i, j] * h[im1, j+1]
                      + ch[8, i, j] * h[ip1, j-1]
                      + ch[9, i, j] * h[im1, j-1]
                      - rhs[i, j]) / (-ch[1, i, j])
        h[i, j] = (1e0 - rlx) * h[i, j] + rlx * hstar
      end
    end

    for l in 1:3
      hfill(dtime, h, sums)
      #@ccall "./PSOM_LIB.so".hfill_(Ref(dtime)::Ref{rc_kind}, pointer(h)::Ptr{rc_kind})::Cvoid
    end

    local maxres = 0e0

    for j in 1:NJ
      local i = 1
      local res = rhs[i,j] - ( 
        ch[1, i, j] * h[i, j]
      + ch[2, i, j] * h[i+1, j]
      + ch[3, i, j] * h[NI, j] 
      + ch[4, i, j] * h[i, j+1] 
      + ch[5, i, j] * h[i, j-1] 
      + ch[6, i, j] * h[i+1, j+1] 
      + ch[7, i, j] * h[NI, j+1] 
      + ch[8, i, j] * h[i+1, j-1] 
      + ch[9, i, j] * h[NI, j-1])
      if (abs(res) > maxres)
        maxres = abs(res)
      end
      for i in 2:NI-1
        local res = rhs[i,j] - ( 
          ch[1, i, j] * h[i, j]
        + ch[2, i, j] * h[i+1, j]
        + ch[3, i, j] * h[i-1, j] 
        + ch[4, i, j] * h[i, j+1] 
        + ch[5, i, j] * h[i, j-1] 
        + ch[6, i, j] * h[i+1, j+1] 
        + ch[7, i, j] * h[i-1, j+1] 
        + ch[8, i, j] * h[i+1, j-1] 
        + ch[9, i, j] * h[i-1, j-1])
        if (abs(res) > maxres)
          maxres = abs(res)
        end
      end
      local i = NI
      local res = rhs[i,j] - ( 
        ch[1, i, j] * h[i, j]
      + ch[2, i, j] * h[1, j]
      + ch[3, i, j] * h[i-1, j] 
      + ch[4, i, j] * h[i, j+1] 
      + ch[5, i, j] * h[i, j-1] 
      + ch[6, i, j] * h[1, j+1] 
      + ch[7, i, j] * h[i-1, j+1] 
      + ch[8, i, j] * h[1, j-1] 
      + ch[9, i, j] * h[i-1, j-1])
      if (abs(res) > maxres)
        maxres = abs(res)
      end
      if (maxres > 3000)
        println("hsolve.jl: STOP. res too large, i,j,maxres=",i,",",j,",",maxres)
        exit(1)
      end
    end


    if (maxres < tol)
      break
    end
  end

  for l in 1:1
    mprove(h, ch, rhs, dtime)
    #@ccall "./PSOM_LIB.so".mprove_(pointer(h)::Ptr{rc_kind}, pointer(ch)::Ptr{rc_kind}, pointer(rhs)::Ptr{rc_kind}, Ref(dtime)::Ref{rc_kind})::Cvoid
  end

end