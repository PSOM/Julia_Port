#!/bin/bash
# Compare a GPU (or other) backend against the CPU backend of the Julia model, without the Fortran
# reference: each case is run on both backends, dumping the full state every step, and compared.
#
# Usage: validation/backend_check.sh <backend> [case ...]     backend: cuda | amdgpu | opencl | jlarray
# Cases as in regress.sh: wiggle (10 steps), sche, hydro (wiggle variants), waves (16x192x96, 3 steps).
# Expected: round-off differences (GPU compilers fuse multiply-adds), ~1e-10 relative or less;
# hxn in waves is a noise-only field (~1e-7 relative, ~1e-17 absolute).
set -e
here="$(cd "$(dirname "$0")" && pwd)"
export PATH="$HOME/.juliaup/bin:$PATH"
be="$1"; shift
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
    for b in cpu "$be"; do
        PSOM_BACKEND=$b bash "$here/run_case.sh" $exp $n --no-fortran "${args[@]}" --tag "bk${tag:+_$tag}_$b" > /dev/null \
            || { echo "$c: $b run FAILED (see $here/runs/${exp}${sfx}_bk*_$b/julia/log.txt)"; exit 1; }
    done
    d0="$here/runs/$exp${sfx}_bk${tag:+_$tag}_cpu"; d1="$here/runs/$exp${sfx}_bk${tag:+_$tag}_$be"
    read NI NJ NK ntr <<< "$(awk '{print $1,$2,$3,$4}' "$d0/dims.txt")"
    echo "== $c: $be vs cpu"
    JULIA_PROJECT="$here" julia "$here/compare.jl" "$d0/julia" "$d1/julia" $NI $NJ $NK $ntr | sed -n '/^worst/,$p'
    paste <(grep kinetic "$d0/julia/log.txt" | awk '{print $NF}') <(grep kinetic "$d1/julia/log.txt" | awk '{print $NF}') \
        | awk '{d=$1-$2; if (d<0) d=-d; r=d/$1; if (r>m) m=r} END{printf "  KE max rel diff %.2e\n", m}'
done
