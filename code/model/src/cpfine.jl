# cpfine: coefficients cpf (19-point stencil) and rhs fn of the pressure equation on the finest
# multigrid level. pbc does the boundary points (j=1, NJ, k=1, NK); this kernel the interior.
@kernel function k_cpfine!(cpf, fn, cxf, cyf, czf, skfc, sifc, sjfc, gqi, gqj, gqk, gqi3, gqj3, edt, lo)
    i, j, k = @index(Global, NTuple) .+ lo
    @inbounds begin
        fn[i, j, k] = edt * (cxf[i, j, k] - cxf[i-1, j, k] + cyf[i, j, k] - cyf[i, j-1, k] + czf[i, j, k] - czf[i, j, k-1]) -
                      (skfc[i, j, k] - skfc[i, j, k-1] + (sifc[i, j, k] - sifc[i-1, j, k] + sjfc[i, j, k] - sjfc[i, j-1, k]))
        cpf[1, i, j, k] = -(gqi[i-1, j, k, 1] + gqi[i, j, k, 1] + gqj[i, j-1, k, 2] + gqj[i, j, k, 2] + gqk[i, j, k-1, 3] + gqk[i, j, k, 3])
        cpf[2, i, j, k] = gqi[i, j, k, 1] + 0.25e0 * (gqj[i, j, k, 1] - gqj[i, j-1, k, 1] + gqk[i, j, k, 1] - gqk[i, j, k-1, 1])
        cpf[3, i, j, k] = gqj[i, j, k, 2] + 0.25e0 * (gqi[i, j, k, 2] - gqi[i-1, j, k, 2] + gqk[i, j, k, 2] - gqk[i, j, k-1, 2])
        cpf[4, i, j, k] = gqi[i-1, j, k, 1] + 0.25e0 * (-gqj[i, j, k, 1] + gqj[i, j-1, k, 1] - gqk[i, j, k, 1] + gqk[i, j, k-1, 1])
        cpf[5, i, j, k] = gqj[i, j-1, k, 2] + 0.25e0 * (-gqi[i, j, k, 2] + gqi[i-1, j, k, 2] - gqk[i, j, k, 2] + gqk[i, j, k-1, 2])
        cpf[6, i, j, k] = gqk[i, j, k, 3] + 0.25e0 * (gqi3[i, j, k] - gqi3[i-1, j, k] + gqj3[i, j, k] - gqj3[i, j-1, k])
        cpf[7, i, j, k] = gqk[i, j, k-1, 3] + 0.25e0 * (- gqi3[i, j, k] + gqi3[i-1, j, k] - gqj3[i, j, k] + gqj3[i, j-1, k])
        cpf[8, i, j, k] = 0.25e0 * (-gqi[i-1, j, k, 2] - gqj[i, j, k, 1])
        cpf[9, i, j, k] = 0.25e0 * (gqi[i-1, j, k, 2] + gqj[i, j-1, k, 1])
        cpf[10, i, j, k] = 0.25e0 * (-gqi[i, j, k, 2] - gqj[i, j-1, k, 1])
        cpf[11, i, j, k] = 0.25e0 * (gqi[i, j, k, 2] + gqj[i, j, k, 1])
        cpf[12, i, j, k] = 0.25e0 * (gqi3[i-1, j, k] + gqk[i, j, k-1, 1])
        cpf[13, i, j, k] = 0.25e0 * (-gqi3[i, j, k] - gqk[i, j, k-1, 1])
        cpf[14, i, j, k] = 0.25e0 * (gqi3[i, j, k] + gqk[i, j, k, 1])
        cpf[15, i, j, k] = 0.25e0 * (-gqi3[i-1, j, k] - gqk[i, j, k, 1])
        cpf[16, i, j, k] = 0.25e0 * (gqj3[i, j-1, k] + gqk[i, j, k-1, 2])
        cpf[17, i, j, k] = 0.25e0 * (-gqj3[i, j, k] - gqk[i, j, k-1, 2])
        cpf[18, i, j, k] = 0.25e0 * (gqj3[i, j, k] + gqk[i, j, k, 2])
        cpf[19, i, j, k] = 0.25e0 * (-gqj3[i, j-1, k] - gqk[i, j, k, 2])
    end
end

# cpf, fn are the flat multigrid storage vectors; the work is done on 4-D/3-D views of their
# first-level part.
function cpfine(dtimel, cpf, fn, S=DEV)
    cpfine_level1!(dtimel, reshape(view(cpf, 1:(19*NI*NJ*NK)), 19, NI, NJ, NK), reshape(view(fn, 1:(NI*NJ*NK)), NI, NJ, NK), S)
end

function cpfine_level1!(dtimel, cpf, fn, S)
    local edt = EPS / dtimel
    pbc(cpf, fn, dtimel, S)
    launch!(S, k_cpfine!, (1:NI, 2:NJ-1, 2:NK-1), cpf, fn, S.cxf, S.cyf, S.czf, S.skfc, S.sifc, S.sjfc,
            S.gqi, S.gqj, S.gqk, S.gqi3, S.gqj3, edt)
end
