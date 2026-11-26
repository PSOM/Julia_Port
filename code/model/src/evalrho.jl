using .cppdefs

function evalrho(rhonew, nl, S=DEV)
  @static if (cppdefs.rhoonly)
    evalrho_rho(rhonew, nl, S)
  else
    evalrho_sT(rhonew, nl, S)
  end
end
