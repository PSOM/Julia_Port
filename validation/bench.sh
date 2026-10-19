#!/bin/bash
# CPU time per step, Fortran reference vs Julia.
#
# Usage: [PSOM_SIZE="NI NJ NK ngrid"] validation/bench.sh <experiment> [nsteps=10] [julia_threads=1]
# Fortran: (time for nsteps+1 steps - time for 1 step) / nsteps, single-threaded, no output.
# Needs runs/<exp>[_size]/{fortran,julia}/ namelists from run_case.sh.

set -e
here="$(cd "$(dirname "$0")" && pwd)"
expe="$1"; n="${2:-10}"; th="${3:-1}"
export PATH="$HOME/.juliaup/bin:$PATH"
source "$here/size_override.sh"; sfx=$(size_suffix)
rdir="$here/runs/$expe$sfx/bench_fortran"; mkdir -p "$rdir"

ftime() {  # $1 = nsteps
    sed -E -e "s#^( *)nsteps *= *[0-9]+#\1nsteps=$1#" -e 's#^( *)pickup_int *= *[0-9]+#\1pickup_int=1000000#' \
        "$here/runs/$expe$sfx/fortran/namelist" > "$rdir/namelist"
    ( cd "$rdir" && /usr/bin/time -p "$here/build/$expe$sfx/nh_ref" < namelist > log.txt 2> time.txt; awk '/^real/{print $2}' time.txt )
}
t1=$(ftime 1); tn=$(ftime $((n+1)))
awk -v a="$t1" -v b="$tn" -v n="$n" -v e="$expe$sfx" 'BEGIN{printf "fortran %s  per step: %.4f s  (init+1 step %.2f s)\n", e, (b-a)/n, a}'

julia -t"$th" --project="$here" "$here/bench.jl" "$expe" "$n"
