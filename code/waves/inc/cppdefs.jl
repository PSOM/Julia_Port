module cppdefs

const runtracmass=false
const periodic_ew=true
const  periodic_ns=false
const  allow_particle=false
const  rhoonly=true
const  relaxation=false
const  fixed_bottom_thickness=false
const file_output=true
const file_output_cdf=true
const file_output_bin=false
const gotm_call=false
const  implicit=false
const parallel=true
# gotm_im selects the implicit vertical-mixing branch in waves/src/advection_and_mixing.
# The Fortran tests "#ifdef gotm_imp", which waves/inc/cppdefs.h never defines, so the Fortran
# waves runs the explicit branch. Set to false to match it.
const gotm_im=false

end