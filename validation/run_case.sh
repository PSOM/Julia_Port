#!/bin/bash
# Run one experiment with both the Fortran reference and the Julia port, then compare full states.
#
# Usage: validation/run_case.sh <experiment> [nsteps] [--no-fortran] [--no-julia] [--tag name] [--set key=value ...]
#   experiment: model | wiggle | ...   (Julia namelist: code/<exp>/namelist*.yml, Fortran: code/<exp>/namelist*)
#   --set key=value : override a PARAM/USER namelist entry in both namelists (true/false for logicals)
#   --tag name      : results go to runs/<experiment>_<name> (default: runs/<experiment>)
#
# Results: validation/runs/<experiment>[_tag]/{fortran,julia}/ and compare.txt
# Every step's full state is dumped (pickup_int=1); out2d/out3d output is disabled.

set -e
here="$(cd "$(dirname "$0")" && pwd)"
code="$here/../code"
expe="$1"; nsteps="${2:-10}"
shift; [ $# -gt 0 ] && shift
run_f=1; run_j=1; tag=""; sets=()
while [ $# -gt 0 ]; do
    case "$1" in
        --no-fortran) run_f=0 ;;
        --no-julia) run_j=0 ;;
        --tag) tag="_$2"; shift ;;
        --set) sets+=("$2"); shift ;;
    esac
    shift
done
# sed scripts applying the --set overrides
sed_f=(); sed_j=()
for kv in "${sets[@]}"; do
    k="${kv%%=*}"; v="${kv#*=}"
    fv="$v"; [ "$v" = true ] && fv=".TRUE."; [ "$v" = false ] && fv=".FALSE."
    sed_f+=(-e "s#^( *)$k *=[^,/]*#\1$k=$fv#I")
    sed_j+=(-e "s#^( *)$k:.*#\1$k: $v#")
done
export PATH="$HOME/.juliaup/bin:$PATH"
source "$here/size_override.sh"   # optional PSOM_SIZE grid override (see size_override.sh)
sfx=$(size_suffix)

if [ "$expe" = model ]; then
    nml_f="$code/namelist"; nml_j="$code/namelist.yml"
else
    nml_f=$(ls "$code/$expe"/namelist* | grep -v '\.yml$' | head -1)
    nml_j=$(ls "$code/$expe"/namelist*.yml | head -1)
fi
rdir="$here/runs/$expe$sfx$tag"
mkdir -p "$rdir/fortran" "$rdir/julia"

if [ $run_f = 1 ]; then
    [ -x "$here/build/$expe$sfx/nh_ref" ] || bash "$here/build_fortran_ref.sh" "$expe"
    rm -f "$rdir"/fortran/op.*
    sed -E -e 's#^( *)dirout *=.*#\1dirout="./",#' -e 's#^( *)pickup_int *= *[0-9]+#\1pickup_int=1#' \
           -e 's#^( *)out2d_int *= *[0-9]+#\1out2d_int=100000#' -e 's#^( *)out3d_int *= *[0-9]+#\1out3d_int=100000#' \
           -e "s#^( *)nsteps *= *[0-9]+#\1nsteps=$nsteps#" "${sed_f[@]}" "$nml_f" > "$rdir/fortran/namelist"
    (cd "$rdir/fortran" && "$here/build/$expe$sfx/nh_ref" < namelist > log.txt 2>&1)
    echo "fortran done: $(grep -c 'kinetic' "$rdir/fortran/log.txt") KE lines"
fi

if [ $run_j = 1 ]; then
    rm -f "$rdir"/julia/op.*
    sed -E -e "s#^( *)dirout:.*#\1dirout: \"$rdir/julia/\"#" -e 's#^( *)pickup_int:.*#\1pickup_int: 1#' \
           -e 's#^( *)out2d_int:.*#\1out2d_int: 100000#' -e 's#^( *)out3d_int:.*#\1out3d_int: 100000#' \
           -e "s#^( *)nsteps:.*#\1nsteps: $nsteps#" "${sed_j[@]}" "$nml_j" > "$rdir/julia/namelist.yml"
    (cd "$code" && sh tools/compile_julia.sh "$expe" > /dev/null)
    enddir=nh_$expe; [ "$expe" = model ] && enddir=nh
    # same flags as the Fortran reference build: no netcdf output (it calls diagnostics that
    # the Fortran reference never runs)
    sed -i '' -E 's/^( *const +file_output_cdf *=).*/\1false/' "$code/exe/$enddir/inc/cppdefs.jl"
    sed -i '' -E 's/^( *const +file_output(_bin)? *=).*/\1true/' "$code/exe/$enddir/inc/cppdefs.jl"
    [ -n "$PSOM_SIZE" ] && write_size_jl "$code/exe/$enddir/inc/size.jl"
    (cd "$code/exe/$enddir" && JULIA_PROJECT="$here" julia -t1 -- src/main.jl "$rdir/julia/namelist.yml" > "$rdir/julia/log.txt" 2>&1) \
        || { echo "julia FAILED, tail of log:"; tail -20 "$rdir/julia/log.txt"; exit 1; }
    echo "julia done: $(grep -c 'kinetic' "$rdir/julia/log.txt") KE lines"
fi

size_jl="$code/model/inc/size.jl"; [ -f "$code/$expe/inc/size.jl" ] && size_jl="$code/$expe/inc/size.jl"
read NI NJ NK ntr <<< $(awk -F= '/const NI/{ni=$2} /const NJ/{nj=$2} /const NK/{nk=$2} /const ntr/{nt=$2} END{print ni+0, nj+0, nk+0, nt+0}' "$size_jl")
[ -n "$PSOM_SIZE" ] && read NI NJ NK <<< "$(echo $PSOM_SIZE | awk '{print $1, $2, $3}')"
echo "$NI $NJ $NK $ntr" > "$rdir/dims.txt"
JULIA_PROJECT="$here" julia "$here/compare.jl" "$rdir/fortran" "$rdir/julia" $NI $NJ $NK $ntr > "$rdir/compare.txt"
paste <(grep kinetic "$rdir/fortran/log.txt" | awk '{print $NF}') <(grep kinetic "$rdir/julia/log.txt" | awk '{print $NF}') \
    | awk '{d=$1-$2; if (d<0) d=-d; printf "KE step %2d  F=%.15e  J=%.15e  rel=%.2e\n", NR-1, $1, $2, d/$1}' > "$rdir/ke.txt"
cat "$rdir/ke.txt"
sed -n '/^worst/,$p' "$rdir/compare.txt"
echo "details: $rdir/compare.txt"
