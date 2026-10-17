function newcor(dtimel, n, S=DEV)
  dfill!(S, S.si, 0e0, 1:NI, 1:NJ, 1:NK)
  dfill!(S, S.sj, 0e0, 1:NI, 1:NJ, 1:NK)
  dfill!(S, S.sk, 0e0, 1:NI, 1:NJ, 1:NK)
end
