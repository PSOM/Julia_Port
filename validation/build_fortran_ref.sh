#!/bin/bash
# Build the Fortran PSOM as a reference executable for validating the Julia port.
#
# Usage: [PSOM_SIZE="NI NJ NK ngrid"] validation/build_fortran_ref.sh [experiment]     (default: model)
#
# - Sources: code/model/{src,inc}, overlaid by code/<experiment>/{src,inc} (same rule as compile_julia.sh).
# - cppdefs.h is regenerated to match the Julia flags in code/model/inc/cppdefs.jl, except:
#     file_output_cdf off (no Fortran netcdf here; the reference uses the binary pickup dumps)
#     parallel off      (the OpenMP linerelax is a racy Gauss-Seidel -> non-deterministic)
#     file_output, file_output_bin on (the binary pickup dumps are what the comparison reads)
# - Flags follow the gfortran reference in code/tools/_values_reference
#   (-fdefault-real-8 promotes the few single-precision locals to Float64, like the Julia code).
#
# Output: validation/build/<experiment>/nh_ref

set -e
here="$(cd "$(dirname "$0")" && pwd)"
code="$here/../code"
expe="${1:-model}"
source "$here/size_override.sh"
bdir="$here/build/$expe$(size_suffix)${BUILD_SUFFIX:-}"

rm -rf "$bdir"; mkdir -p "$bdir"
cp "$code"/model/src/*.f90 "$code"/model/inc/*.h "$code"/model/inc/*.inc "$bdir"/
if [ "$expe" != "model" ]; then
    cp "$code/$expe"/src/*.f90 "$bdir"/ 2>/dev/null || true
    cp "$code/$expe"/inc/*.h   "$bdir"/ 2>/dev/null || true
fi
[ -n "$PSOM_SIZE" ] && write_size_h "$bdir/size.h"     # validation grid-size override
# FIX_LINERELAX_WRAP=1: patch the Fortran linerelax index wrap (i>nxm -> i-nxm only once, which goes
# out of bounds on coarse grids when NI/2 > nxm) to i=mod(i-1,nxm)+1, i.e. what the Julia port does.
# Validation-only variant to show that this Fortran bug is the only remaining difference.
if [ "${FIX_LINERELAX_WRAP:-0}" = 1 ]; then
    sed -i '' -E 's/if \(i\.gt\.nxm\) i = i -nxm/i = mod(i-1,nxm)+1/' "$bdir/linerelax_periodicew.f90"
    grep -q 'mod(i-1,nxm)' "$bdir/linerelax_periodicew.f90" || { echo "linerelax patch failed"; exit 1; }
fi

# cppdefs.h derived from the Julia cppdefs.jl (true -> #define, false -> #undef)
jl_cppdefs="$code/model/inc/cppdefs.jl"
[ -f "$code/$expe/inc/cppdefs.jl" ] && jl_cppdefs="$code/$expe/inc/cppdefs.jl"
awk '$1=="const" { split($0, a, "="); name=a[1]; sub(/^[ \t]*const[ \t]+/, "", name); gsub(/[ \t]/, "", name);
                     val=a[2]; gsub(/[ \t]/, "", val);
                     if (name=="file_output_cdf" || name=="parallel" || name=="gotm_call") val="false";
                     if (name=="file_output" || name=="file_output_bin") val="true";
                     if (val=="true") print "#define " name; else print "#undef  " name }' "$jl_cppdefs" > "$bdir/cppdefs.h"

# Object list = lobj_general of code/optfile (gotm and netcdf objects excluded)
objs=$(grep '^lobj_general=' "$code/optfile" | sed 's/^lobj_general=//')
# experiment-specific objects (code/<exp>/optfile_add), without the netcdf writers
if [ -f "$code/$expe/optfile_add" ]; then
    objs="$objs $(grep '^lobj_additional=' "$code/$expe/optfile_add" | sed 's/^lobj_additional=//' | tr ' ' '\n' | grep -v '^write_cdf' | tr '\n' ' ')"
fi

FC=${FC:-gfortran}
# anaconda ships an old ld that cannot read the macOS SDK: force the system linker
[ "$(uname)" = Darwin ] && LDFLAGS=${LDFLAGS:-"-B/usr/bin/"}
FFLAGS=${FFLAGS:-"-O2 -cpp -fdefault-real-8 -fdefault-double-8 -ffree-line-length-none -ffixed-line-length-none -fno-range-check -w"}

cd "$bdir"
# header (module) first, then write_bin_mod (module used by others), then the rest
$FC $FFLAGS -c header.f90
$FC $FFLAGS -c write_bin_mod.f90
for o in $objs; do
    f="${o%.o}.f90"
    case "$f" in header.f90|write_bin_mod.f90) continue ;; esac
    $FC $FFLAGS -c "$f"
done
# Shared library for routine-level differential tests (validation/difftest.jl).
# ref_read_params() = the namelist-reading part of the Fortran main program (ini_param.h),
# reading ./namelist instead of stdin.
sed -e 's/READ *(\*, *PARAM)/OPEN(77,file="namelist",status="old"); READ (77,PARAM)/' \
    -e 's/READ *(\*,/READ (77,/' ini_param.h > ini_param_lib.h
cat > ref_shim.f90 <<'EOF'
subroutine ref_read_params()
#include "cppdefs.h"
  USE header
  IMPLICIT NONE
#include "ini_param_lib.h"
  CLOSE(77)
end subroutine ref_read_params

! OpenMP is off in the reference build; some experiment copies (e.g. waves linerelax) call this
! without the "#ifdef parallel" guard.
integer function omp_get_num_threads()
  omp_get_num_threads = 1
end function omp_get_num_threads
EOF
$FC $FFLAGS -c ref_shim.f90
$FC $FFLAGS $LDFLAGS -o nh_ref *.o
echo "built $bdir/nh_ref"


libext=so; [ "$(uname)" = Darwin ] && libext=dylib
$FC $FFLAGS $LDFLAGS -shared -o "libpsom_ref.$libext" $(ls *.o | grep -v '^main\.o$')
echo "built $bdir/libpsom_ref.$libext"
