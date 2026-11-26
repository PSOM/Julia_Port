#!/bin/bash
# Regression check for the GPU refactor: run the Julia model on the standard test cases and compare
# the full state, every step, with a saved Julia baseline (expected: bit-identical on the CPU backend)
# and with the Fortran reference.
#
# Usage: validation/regress.sh save            # store the current Julia results as the baseline
#        validation/regress.sh [check] [case...] # rerun Julia and compare (default: all cases)
# Cases: wiggle (10 steps), sche (wiggle, Shchepetkin, 10), hydro (wiggle, fnhhy=0, 10), waves (16x192x96, 3)
# Env: PSOM_BACKEND (cpu | jlarray | cuda | amdgpu) is passed through to the Julia runs.
#      REGRESS_BASELINE: name of the baseline directory (default julia_baseline).
#      PSOM_THREADS: Julia threads for the runs (default 1).
set -e
base="${REGRESS_BASELINE:-julia_baseline}"
here="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
mode="${1:-check}"; [ $# -gt 0 ] && shift
cases=("$@"); [ ${#cases[@]} -eq 0 ] && cases=(wiggle sche hydro waves)
ulimit -s 65520 2>/dev/null || true
for c in "${cases[@]}"; do
    unset PSOM_SIZE; args=(); exp=wiggle; n=10; tag=""
    case $c in
        wiggle) ;;
        sche) args=(--set use_Shchepetkin=true); tag=sche ;;
        hydro) args=(--set fnhhy=0e0); tag=hydro ;;
        waves) exp=waves; n=3; export PSOM_SIZE="16 192 96 4" ;;
    esac
    sfx=""; [ -n "$PSOM_SIZE" ] && sfx=$(source "$here/size_override.sh"; size_suffix)
    rdir="$here/runs/$exp$sfx${tag:+_$tag}"
    targs=(); [ -n "$tag" ] && targs=(--tag $tag)
    if [ "$mode" = save ]; then
        bash "$here/run_case.sh" $exp $n "${args[@]}" "${targs[@]}" > /dev/null || { echo "$c: run FAILED (see $rdir/*/log.txt)"; exit 1; }
        rm -rf "$rdir/$base"; cp -r "$rdir/julia" "$rdir/$base"
        echo "$c: baseline saved ($rdir/$base)"
    else
        bash "$here/run_case.sh" $exp $n --no-fortran "${args[@]}" "${targs[@]}" > /dev/null \
            || { echo "$c: julia FAILED, tail of $rdir/julia/log.txt:"; tail -15 "$rdir/julia/log.txt"; exit 1; }
        read NI NJ NK ntr <<< "$(awk '{print $1,$2,$3,$4}' "$rdir/dims.txt")"
        b=$(JULIA_PROJECT="$here" julia "$here/compare.jl" "$rdir/$base" "$rdir/julia" $NI $NJ $NK $ntr | sed -n '/^worst/,$p' | awk 'NR>1{if ($2+0>m) {m=$2+0; f=$1}} END{printf "%.3e (%s)", m, f}')
        f=$(sed -n '/^worst/,$p' "$rdir/compare.txt" | awk 'NR>1{if ($2+0>m) {m=$2+0; f=$1}} END{printf "%.3e (%s)", m, f}')
        echo "$c: vs baseline max rel diff $b | vs Fortran $f"
    fi
done
