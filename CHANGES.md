# Changes on branch `fix-and-validate` (October 2026)

This branch makes the Julia port of PSOM reproduce the Fortran model, and then ports its time step to
GPUs. Related documents in this directory:
- `VALIDATION.md`: the tests used to compare Julia with Fortran, how to run them, and current results.
- `GPU_PORTING.md`: the GPU design, the porting rules, and how to run on a GPU server.
- `PLAN.md`: the work plan, with what is done and what is still open.
- `TODO.md`: what is left to finish the GPU port.

The Fortran copies in `code/` were also fixed where the Fortran itself was wrong. The same Fortran fixes
were made in `PSOM/V1.0` (branch `fix-fortran-bugs`, see its `CHANGES.md`).

---

## Summary

**Before** (branch `master`, January 2024), the pure-Julia model ran wiggle, but did not reproduce the
Fortran:
- the kinetic energy (KE) differed by 4e-7 (relative) already at step 0 and the fields by up to 3e-4;
- passive tracers were never time-stepped (100% wrong);
- several solver parts were wrong (they converged anyway, more slowly, or only mattered in some cases);
- the `waves` experiment didn't run at all.

**Now** Julia and Fortran agree to floating-point round-off in every case tested (wiggle in its default,
Shchepetkin and hydrostatic configurations, and waves): KE to ≤1e-13 relative and fields to ≤2e-10, over 3
to 100 steps. That is the level at which two correct codes differ only in the order of their arithmetic
(see `VALIDATION.md`).

**Changes that affect the model output**, compared with `master`:

| Change | Section | Effect on the Julia output |
|--------|---------|----------------------------|
| The Julia translation bugs (18 entries) | 1 | Julia now matches the Fortran (it didn't before) |
| waves made to run | 2 | waves runs and matches the Fortran |
| Fortran bug F5 (`staticsigma`, also in the Julia) | 3 | Non-hydrostatic runs: initial state changes slightly (wiggle KE(0) by 4.5e-7 rel.); both codes |
| Fortran bug F1 (`linerelax` wrap) | 3 | Fortran only: it now matches what Julia always did |
| D1 smooth waves sponge | 4 | waves: different near the north and south sponges; both codes |
| D2 `mldepth` from the namelist | 4 | Diagnostics only |
| D3 multi-colour solvers | 4 | All runs, at solver-tolerance level (wiggle KE 1.7e-9 rel. after 5 steps); both codes; makes OpenMP deterministic and GPUs possible |
| GPU port | 6 | None on the CPU (bit-identical), apart from one reordered sum (`hbc`, ≤8e-11); round-off on GPUs |

---

## 1. Julia translation bugs (core model, `code/model/src/`)

### Where they came from

Reading the history of `master`:
- **Leftover index shifts.** The port first stored the Fortran arrays as ordinary 1-based Julia arrays, so
  that Fortran's `a(i,j,k)` with indices starting at 0 became `a[i+1,j+1,k+1]`. On 28–29 December 2023
  the arrays were switched to `OffsetArray`s, which keep the Fortran index ranges, so every `+1` should
  have been removed. Some were not. Examples: `uvchy.jl` (written on 1 December with `wy[i+1,j+1,k+1]`;
  `wy` became an OffsetArray on 28 December), `calcskfc.jl` (`ux[i+1,j+1]`), `intpol_3rdorder.jl`
  (`cy[i0+1,j+1,k+1]`), `srcface_3rdorder.jl` (`bbj[i,j+1]`), and probably the `(i+1,j+1)` terms in
  `rpevalgrad_Song.jl`.
- **Transcription slips** while translating line by line: `x+.5` became `x+5e0` (`prolong`),
  `do i=1,nxm` became `for i in nxm` (`sor`), `vfbcs` was used for `vfbcn` (`hfill`), a missing `rhs -`
  (`mprove`), a missing `edt*` and mis-grouped terms (`cpfine`, `pbc`), shifted level indices
  (`rpevalgrad_Song`, `rpevalgrad_Sche`, `cdiv`), reversed array order (`diag_n2budget`).
- **Structure.** The tracer update was placed inside the `if selvar == 5` branch instead of after it
  (`advection_and_mixing`). In `dgtsl`, the pivot swap assigned to `p`, which in Julia is the global
  pressure array, instead of the routine's right-hand side `b`.
- **Not ported.** The Fortran random number generator `ran3`, whose state carries over between calls,
  was replaced by Julia's `Random.seed!` on every call, and its seeding call in `ini_st` was commented
  out. The multigrid fallback path (after 30 unconverged cycles) used names that didn't exist.

The port was originally developed inside the Fortran executable, replacing one Fortran routine at a time
(the Julia arrays shared the Fortran memory through `@fortGetArr`). The stand-alone Julia mode was added
on 23 January 2024 as "experimental" and, as far as the history shows, never compared with the Fortran
as a whole.

### How they were found

`validation/difftest.jl` (see `VALIDATION.md`) runs every routine of the time step in Julia and in the
Fortran on identical inputs and compares all ~190 shared model variables after each one. A routine whose
output differed was a bug in that routine; each fix was accepted only when the routine matched to ≤1e-12
and the whole-model comparison improved. Some bugs (`intpol`, `srcface`, `calcskfc`, `conadjust`, `cdiv`)
were found by reading the code: they don't change the wiggle and waves tests (flat bottom, rectangular
grid, no convection, or a diagnostic), so the tests couldn't see them.

### The bugs

| # | Commit | File | What was wrong | Fortran | Effect of the bug |
|---|--------|------|----------------|---------|-------------------|
| 1 | `c591616` | `rpevalgrad_Song.jl` | (a) `zb = DL*zf[i,j,k]`; (b) the x-halo of `rp` filled for `j=1:NJ` only; (c) `Jy2` written at level `k+1`; (d) `(i+1,j+1,k+1)` in the k=0 `Jx2`/`dzsig_x` terms | (a) `zf(i,j,k-1)`; (b) `j=0:NJ+1`; (c) `Jy2(j,k)`; (d) `(i+1,j,k+1)` | Wrong baroclinic pressure gradient, hence wrong initial geostrophic velocities. **The main cause of the KE disagreement** (section 1.1). |
| 2 | `db4f538` | `advection_and_mixing.jl` | Tracer update nested inside `if selvar == 5` | separate loop | **Passive tracers never time-stepped.** |
| 3 | `58ee204` | `uvchy.jl` | `wy[i+1,j+1,k+1]` | `wy(i,j,k)` | Wrong vertical-velocity predictor where sigma levels slope. |
| 4 | `3affc0d` | `hfill.jl`, `hsolve.jl` | North boundary used the south flux sum `vfbcs`; column sums with Julia's `sum` | `vfbcn`; sequential sum | Wrong free surface `h` at the north boundary when boundary fluxes are non-zero. |
| 5 | `bb36c5f` | `mprove.jl` | Residual computed as `A*he` | `rhs - A*he` | The convergence test never passed: always 1000 sweeps, and a slightly different `h`. |
| 6 | `1c061c4` | `cpfine.jl` | RHS `edt*(div - Δskfc + Δsifc + Δsjfc)`; `cpf[6]`, `cpf[7]` used shifted `gqj3` indices (and a wrong sign in `cpf[7]`) | `edt*div - (Δskfc + (Δsifc+Δsjfc))` | Wrong non-hydrostatic pressure equation. |
| 7 | `bb6ad54` | `pbc.jl` | `fn -= wfbcb` (no `edt`) | `fn - edt*wfbcb` | Wrong bottom-flux term (zero in wiggle). |
| 8 | `ce6a497` | `sor.jl` | `for i in nxm` (only i=nxm); `cp[2]*p[i,j-1,k-1]` | `do i=1,nxm`; `p(i+1,j,k)` | The coarsest-grid multigrid smoother did almost nothing. |
| 9 | `6f2bcfc` | `prolong.jl` | `x = x + 5e0`, so all interpolation weights were wrong | `x+.5` | **Wrong multigrid interpolation** (it still converged, more slowly). |
| 10 | `266632f` | `dgtsl.jl` | The pivot swap exchanged entries of the global pressure `p` instead of the RHS `b` | swaps `b` | Corrupts `p` whenever the tridiagonal solver pivots. |
| 11 | `87c3af8` | `mgrid.jl` | Fallback after 30 unconverged cycles: `rhs_r` undefined; `mgpfill` given a 1-based view | | Crash whenever multigrid fails to converge. |
| 12 | `d580cdb` | new `ran3.jl`; `wiggle/src/linerelax_periodicew.jl`, `wiggle/src/ini_st.jl` | `Random.seed!(91294)` on every call; the seeding `ran3(44294)` in `ini_st` commented out | Numerical Recipes `ran3`, state kept between calls, seeded in `ini_st` | Different line-relaxation order: 1e-7 per solve, 1e-4 after 10 steps. |
| 13 | `16c2ff3` | `rpevalgrad_Sche.jl` | `rx[0,:] = rx[NI-1,:]` | `rx(NI,:)` | Shchepetkin pressure gradient wrong at the periodic boundary. |
| 14 | `4f1a4f7` | `diag_n2budget.jl` (+ waves copy) | `testdepth = [0.9,0.75,0.5]*mldepth` | `(1)=0.5, (2)=0.75, (3)=0.9` | Diagnostics stored under the wrong index. |
| 15 | `0a7a784` | `diag_n2.jl` | Row mean recomputed inside the i loop (O(NI²) work), with Julia's `sum` | sequential `SUM` | Diagnostics only; now faster and bit-exact. |
| 16 | `25dac70` | `intpol_3rdorder.jl`, `calcskfc.jl`, `srcface_3rdorder.jl` | `cy[i0+1,j+1,k+1]`, `ux[i+1,j+1]`, `bbj[i,j+1]` | `cy(i0,j,k)`, `ux(i,j)`, `bbj(i,j)` | No effect on the tests (rectangular grids); wrong on curvilinear grids. |
| 17 | `18932c1` | `conadjust_sT.jl` | Tracer mixed with itself (`Tr[k]` twice) and copied to `Tr[i,j+1,k]` | mixes `Tr(k)` with `Tr(k+1)`, copies to `Tr(i,j,k+1)` | Wrong tracers wherever convective adjustment happens. s, T, ρ were right. |
| 18 | `38835d9` | `cdiv.jl` | `wz[i,j,k]` in the upper term | `wz(i,j,k+1)` | The printed maximum divergence only. |

"Julia's `sum`" (bugs 4, 15): it adds in a different order from the Fortran loop (pairwise, and the
compiler may reorder it), so the results differ in the last digits. Harmless on its own, but it prevents
bit-exact comparisons, so these sums are now sequential loops as in the Fortran.

### 1.1 Which fix made the Julia agree with the Fortran

To answer this directly, the Julia code at each commit of the fix series was rerun against the original,
unpatched Fortran (both codes as they were before any change on this branch), wiggle, 10 steps. Each row
is the state after that commit, i.e. with that fix and all the ones above it.

Relative differences, Julia vs Fortran: KE at step 0, KE at the worst of steps 0–10, the worst field (largest
relative difference of any field at any step), and the tracer `Tr`.

| State after commit | Fix (table above) | KE, step 0 | KE, worst step | Worst field | `Tr` |
|--------------------|-------------------|-----:|-----:|-----:|-----:|
| `aa2085a` (`master` + comparison tools) | none | 4.4e-7 | 7.4e-7 | 1.0 (`Tr`) | 1.0 |
| `c591616` | 1 `rpevalgrad_Song` | **6.6e-15** | 3.8e-8 | 1.0 (`Tr`) | 1.0 |
| `db4f538` | 2 tracers time-stepped | 6.6e-15 | 3.8e-8 | 3.5e-4 (`wf`) | **9.5e-6** |
| `58ee204` … `ce6a497` | 3–8 | 6.6e-15 | 3.7e-8 | 3.5e-4 (`wf`) | 9.5e-6 |
| `6f2bcfc` | 9 `prolong` weights | 6.6e-15 | **1.1e-8** | 1.7e-4 (`wf`) | 8.0e-6 |
| `266632f`, `87c3af8` | 10, 11 | 6.6e-15 | 1.1e-8 | 1.7e-4 (`wf`) | 8.0e-6 |
| `d580cdb` | 12 `ran3` sequence | 6.6e-15 | **6.6e-15** | **5.8e-11** (`hxn`) | **3.9e-13** |

wiggle with the Shchepetkin pressure gradient: after `d580cdb` KE 5.2e-12, worst field 3.0e-8 (`v`); after
`16c2ff3` (fix 13) KE 3.8e-15, worst field 3.0e-11.

wiggle hydrostatic (`fnhhy=0`): **crashed after step 0** up to and including `6f2bcfc`. The multigrid
pressure solve didn't converge within 30 cycles, and the fallback code then failed (`UndefVarError: rhs_r`,
bug 11). From `266632f` (fix 10, `dgtsl` pivoting, which had been corrupting the pressure `p`) it runs, and
it agrees to round-off: KE ≤3.6e-15, worst field 8.9e-11. (Fix 11 makes the fallback itself work, should it
ever be needed.)

**So, for the kinetic energy:**
- **Step 0** (the initial state, 4.4e-7): fixed by **fix 1, `rpevalgrad_Song`**. The initial velocities are
  geostrophic, computed from the baroclinic pressure gradient, and the wrong `zf` level, `Jy2` level and halo
  rows made that gradient wrong.
- **The later steps** (a difference that appeared from step 1 on and grew to 1e-8 in KE and 1e-4 in `wf`):
  - fix 9, the multigrid interpolation weights in `prolong`, removed about two thirds of it;
  - **fix 12, the `ran3` sequence**, removed the rest. The Fortran line relaxation (`linerelax`) starts each
    sweep at a random line chosen with `ran3`, a generator whose state carries over from call to call. The
    Julia restarted Julia's own generator from the same seed on every call, so the lines were relaxed in a
    different order. Each order is a valid way of converging the pressure solve, but the results differ at the
    solver's tolerance, and those differences grew with the steps. Fix 12 ported `ran3` exactly and restored
    the seeding call in wiggle's `ini_st`. (The multi-colour ordering D3, section 4, later removed the random
    start altogether, in both codes.)
- Fixes 3–8, 10 and 11 don't change these wiggle numbers. The code they fix is inactive in this setup, or its
  effect is too small to see here: flat bottom (so sloping-sigma terms vanish), zero boundary fluxes, and a
  multigrid that converges without needing the fallback or pivoting. They matter in other setups, and each
  was confirmed routine by routine with `difftest.jl`.

These runs used the original Fortran. The Fortran fixes made afterwards (section 3, in particular F5) changed
both codes in the same way, and they still agree to round-off.

---

## 2. The `waves` experiment (`code/waves/`, commit `e91eefe`)

Before this, the Julia waves crashed at load time and then in the first time step:
- `model/inc/includeFile.jl` now includes an optional per-experiment `inc/includeExtra.jl`, which
  `waves/inc/includeExtra.jl` uses for `sponge`, `diag_ptc`, `intpol_pt`, `gradup3df`. Before, files that
  exist only in an experiment were silently never loaded.
- `sponge.jl`: `spl`/`splz` undefined; `rho_refB(j)` used call syntax; `ffc[NI/2,NJ/2]` used float indices.
- `momentum.jl`: passed the `Ref` `dtim` to `sponge` instead of its value.
- `cppdefs.jl`: `gotm_im=true` enabled an implicit-mixing branch that the Fortran compiles out (it tests
  `gotm_imp`, which `waves/inc/cppdefs.h` never defines). Set to `false`, as in the Fortran.
- `header.jl`: `mat_A`…`mat_test` declared unconditionally, as in `header.f90`.
- `advection_and_mixing.jl`: zeroed the undefined `mat_E` instead of `mat_test`, and **overwrote `cz` with the
  tracer tendency on every pass**. The implicit branch now uses the right names, but it isn't validated
  (it is compiled out in the Fortran).
- `wind_stress.jl`: shadowed the header arrays `stress_top_x/y` with scalars, used float ranges (`NJ/2`),
  and rebound the array instead of zeroing it. It now follows the Fortran, including integer `yw`/`yset`.
- `mixing_vertical.jl`: the worker had no `selvar` argument, used `k` before defining it, and wrote `kz`
  instead of `Kz`.
- `intpol_pt.jl`: `cppdefs.periodicew` → `cppdefs.periodic_ew`.

Verified on a 16×192×96 grid: every routine matches in `difftest.jl waves 2 synced`, and the whole model
agrees to round-off (`VALIDATION.md`).

---

## 3. Bugs in the Fortran (fixed in both codes)

These were bugs in the original Fortran. Where the Julia port had copied them faithfully, the Julia was
fixed in the same commit. Each was fixed identically in `PSOM/V1.0` (`fix-fortran-bugs`), where its
`CHANGES.md` has the full description.

| # | V1.0 | here | Fix | Julia changed too? | Effect on results |
|---|------|------|-----|--------------------|-------------------|
| F1 | `c32c53d` | `36fb430` | `linerelax`: periodic wrap of the line start, `i = mod(i-1,nxm)+1` (the Fortran went out of bounds when `NI/2` exceeded a coarse level's size) | no (Julia already used `mod`) | Fortran waves now matches Julia; wiggle unchanged |
| F2 | `8806901` | `75bd552` | OpenMP `REDUCTION(max:maxres)` in `resid`, `hsolve`, `mprove` (a data race) | no | OpenMP only |
| F3 | `f7f59ec` | `7d9566b` | `rpevalgrad_Song`: k=0 entries at the right index (`Jx2(i,0)`, `dzsig(.,0)`, `dzsig_x(.,0)`) | yes (`dzsig_x[i,0]`) | none |
| F4 | `f8ff579` | `610ca7a` | `diag_n2budget`: `avezf(0:NK)` (read out of bounds at k=1) | yes | none in the tests |
| F5 | `025e688` | `5a8ca49` | `staticsigma`: `gi3`/`gj3` loops end at NK-1, so they no longer overwrite the top-layer values from `sigma` with 0 (or garbage) | yes | **Non-hydrostatic runs: initial state changes slightly** (wiggle KE(0) by 4.5e-7 rel., fields up to 3e-3) |
| F6 | `2a25625` | `1f06408` | waves `diag_n2budget`: halo `i2=NI` (was `NK`) | (already right) | diagnostics only |
| F7 | `81cb49c` | `2e38e82` | waves `linerelax`: `#ifdef parallel` around `OMP_GET_NUM_THREADS` | no | build fix |

---

## 4. Model changes decided by the maintainers (both codes)

| # | V1.0 | here | Change | Effect |
|---|------|------|--------|--------|
| D1 | `528f4b8` | `3dca709` | **waves `sponge`: smooth taper.** The Fortran's `cos((j-1)/spl*PI/2)` used integer division, so it was a step function (full strength for j=0..15); now real division. Julia had reproduced the step. | waves: different near the north and south sponges. |
| D2 | `6406683` | `b64dfb2` | **`mldepth` from the namelist**: optional group `&diag mldepth=… /` (YAML: `DIAG: mldepth:`), default 10 m, used by the N² diagnostics. Experiments that build their initial state from `mldepth` (wiggle 50 m, NA, NP_winter) still set it in `ini_st`. | Diagnostics only; the waves N² diagnostics no longer divide by zero. |
| D3 | `10f2919` | `633dc7e` | **Multi-colour Gauss–Seidel/SOR** in `linerelax`, `hsolve`, `mprove`: points are split into colours by the parity of i and j (a third i-colour when the periodic x-size is odd) and the colours are swept in turn. Points of one colour are independent, so they can be updated in parallel. Julia: `ncolours_i`, `colour_irange` in `linerelax_periodicew.jl`; the wiggle `linerelax` override was removed. | All runs, at solver tolerance (wiggle KE after 5 steps: 1.7e-9 rel.). OpenMP runs are now bit-identical to serial runs, and the solvers can run on a GPU. |

**Stack size (Fortran with OpenMP only; not changed).** With OpenMP, the Fortran puts every local array on
the stack, and the full-grid local arrays (e.g. in `sigma_toplayer`, `staticsigma`, `hsolve`) go on the main
thread's stack. A small stack crashes the run at start-up: waves on 16×192×96 needs 64 MB, the full waves grid
about 0.5 GB. The repository's ifort builds avoid this (`-Wl,-stack_size,0x40000000` on macOS; `ulimit -s
unlimited` on clusters). Details in `PSOM/V1.0/CHANGES.md`, section 4. **The Julia port is not affected**:
Julia arrays always live on the heap.

---

## 5. Restructuring and CPU performance

| Commit | Change | Verified |
|--------|--------|----------|
| `dc31525`, `d5f06ed` | `main.jl` split into `initialize()` / `step!()` / `main()`, and runs only when executed as a script. As in the Fortran, the initial output is skipped on pickup. Removed a stray `then`; `ini_param.jl` is included relative to the file. | wiggle unchanged |
| `ee704df` | `mgrid` work arrays allocated once; function barriers in `cpfine`/`vface`; no slice copy in `linerelax` | bit-identical |
| `fcc7362` | `validation/bench.sh`, `bench.jl` | |
| `aa2085a` | `write_bin.jl` also writes the full state in the Fortran raw layout (`op.pickup.<step>.bin`), for the comparisons | |

---

## 6. GPU port

The whole time step now runs as KernelAbstractions.jl kernels, from one source for CPUs and GPUs.
`PSOM_BACKEND=cpu` (default), `cuda` (NVIDIA) or `amdgpu` selects the device. Design, rules and server
instructions: `GPU_PORTING.md`.

- The header arrays stay ordinary host arrays, so initialisation, I/O and diagnostics are unchanged. `DEV`
  is the same set of arrays on the device (on the CPU it *is* the host set, so nothing is copied). The
  state is copied to the device once after initialisation, and back only on output steps.
- Every loop nest of the time step is a kernel. The Fortran index ranges are kept (OffsetArrays work
  inside GPU kernels).
- Sequential parts of the Fortran are kept exact: the coarsest-level `sor` runs as a wavefront over
  i+2j+3k = const, and the `mgpfill` faces as wavefronts over anti-diagonals. The `hfill` ghost rows and two
  `mgpfill` edge rows are single-work-item loops.
- A hidden dependency found along the way: the diagnostic `diag_n2budget` (Fortran too) recomputes `rho`
  and overwrites the ghost points of u, v, w every step, and the next step uses them. On a GPU the
  diagnostics only run on output steps, so these two side effects now run on the device every step.

| Commit | What |
|--------|------|
| `f5316bf` | `validation/regress.sh` (bit-exact Julia baselines); KernelAbstractions, Adapt, JLArrays |
| `4a84b59` | `arch.jl`: backends, `HOST`/`DEV`, `launch!`, work arrays; first kernels (coriolis, newcor, newsrc, pcorrect, evalrho, vcenter, facediv, cdiv, cfdiv) |
| `ce50aa7` | `GPU_PORTING.md`; `validation/check_kernels.jl` |
| `d417f81` | `hbc`: sequential vertical sums, as in the Fortran (the only deliberate arithmetic change of the port, ≤8e-11 rel.; new baseline saved) |
| `5baf7b6` | Free surface: hsolve, chfine, hbc, hfill, mprove |
| `00f4334` | Multigrid: mgrid, cpfine, pbc, cpcors, resid, restrict, prolong, efill, linerelax, sor, mgpfill, vface |
| `bf17da8`…`adccf18` (merge `09f0f1d`) | advect, mixing, advection_and_mixing, heat_flux, wind_stress, conadjust, waves sponge |
| `12a3aca`…`f696ca5` (merge `4f5f048`) | findzall, sigma, intpol, rpevalgrad_Song/Sche, srcface, calcfkfc, vhydro, uvchy |
| `b4a5704`, `1fe2bab` | Time-step glue (device copies, output-step sync, KE on the device); the `diag_n2budget` side effects on the device |
| `75c47fc`, `c74a66c`, `2865140`, `ce109ca` | Benchmarks on GPU backends; the `opencl` test backend; `backend_check.sh`, `gpu_bench.sh`; documentation |

**Checks** (details in `VALIDATION.md`): CPU backend bit-identical to the code before the port, on 1 and 4
threads; emulated GPU arrays bit-identical; kernels compiled by a real GPU compiler (OpenCL) agree to
round-off; every routine still matches the Fortran.

**CPU speed** (Intel i7 MacBook; Fortran gfortran -O2, serial):

| Case | Fortran | Julia on `master` | Julia now, 1 thread | Julia now, 4 threads |
|------|------:|------:|------:|------:|
| wiggle 24³ | 0.047 s | 0.096 s | 0.093 s | 0.079 s |
| waves 16×192×96 (295k cells) | 1.00 s | 1.64 s | 0.93 s | 0.50 s |
| waves 32×384×96 (1.2M cells) | | | | 1.66 s |

**Not done yet:** running on a real NVIDIA/AMD GPU, profiling there, and any solver changes that would
alter results (e.g. convergence checks or a different solver for the free surface). See `PLAN.md`.

---

## 7. Repository and running

| Commit | Change |
|--------|--------|
| `376ae1d` | `includeFile.jl` installs KernelAbstractions and Adapt on first use (as it already did for OffsetArrays, NetCDF, YAML); a clear error if the package for a GPU backend is missing. `code/exe/julia_launch.sh` uses the validation environment (the tested, pinned package versions) unless `JULIA_PROJECT` is set. Checked from an empty environment: with the pinned versions wiggle reproduces the validated results to the last digit. With the latest versions (KernelAbstractions 0.10.1) it runs and agrees to round-off. |
| `6efd1c9` | Removed build products and editor backups: `code/mkfile/Makefile` and the Linux executable `code/mkfile/nh`, the compiled `tools/preproc.exe` and `tools/preprocmine`, `*~`, `*.bak` and stray `.mod` files. `.gitignore` covers them now. |

---

## 8. Known remaining issues (unchanged, in both codes)

- `hsolve` never reaches its tolerance (1e-12) and always runs all 100 iterations; `mprove` then runs
  another 114–433. Results are fine, but it costs time, especially on a GPU.
- `rpevalgrad_Sche` flips the sign of never-written boundary zeros every call (harmless).
- `findzall` uses the undefined `dzbot` under the `fixed_bottom_thickness` option (off).
- The compiled-out implicit/gotm branches of `mixing_vertical` and `advection_and_mixing` use undefined names.
- `mixing_vertical`'s Kz divides by f (infinite on an f=0 plane).
- In wiggle, the computed short-wave heat flux is set to zero afterwards.
- `code/tools/_values_reference` lists wiggle kinetic energies from 2013 (KE(0) = 2.92e-3) that don't
  correspond to the current wiggle setup (KE(0) = 1.02e-3), so they can't be used as a check.
- Experiments other than wiggle and waves (NA, shelfbreak, NP_*, …) have not been compared with the Fortran.
