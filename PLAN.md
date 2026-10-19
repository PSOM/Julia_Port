# PSOM Julia port: fix, validate, then port to GPU

Living document. Status markers: `[ ]` todo, `[~]` in progress, `[x]` done.
What was actually changed, and why, is logged in `CHANGES.md`; the tests are described in `VALIDATION.md`;
the GPU design and conventions are in `GPU_PORTING.md` (all in this directory).

## Ground rules

- **The Fortran model is the reference.** The Fortran in `code/` (the same as `PSOM/V1.0`) is the "truth"
  the Julia code must match. It was only edited where it was itself wrong (bug fixes F1–F7) or where the
  maintainers decided on a change (D1–D3); each such change was made identically in `PSOM/V1.0` (branch
  `fix-fortran-bugs`) and here. See `CHANGES.md`.
- **Work happens on the git branch `fix-and-validate`**, one commit per fix, pushed to GitHub.
- **Nothing counts as "fixed" until it has been checked against Fortran.** Each change gets an entry
  in `CHANGES.md` saying what it was, the Fortran line it matches, and how it was checked
  (tools and tests: `VALIDATION.md`).
- **Hardware limits on the development machine.** It is an Intel Mac with an AMD Radeon Pro 5300M. Julia's
  GPU backends (CUDA.jl for NVIDIA, AMDGPU.jl for ROCm on Linux) cannot run there. GPU code is written with
  KernelAbstractions.jl, which also runs on CPU threads, so correctness is checked locally; speed has to be
  measured on the GPU server.

---

## Phase 0: Setup (done)
- [x] Install Julia (juliaup) and the packages the port uses (OffsetArrays, NetCDF, YAML).
- [x] Create branch `fix-and-validate` in `Julia_Port`.
- [x] Create `Julia_Port/validation/` for all checking tools. Build products and output are gitignored.

## Phase 1: Fortran reference run (done; see CHANGES.md)
Goal: a reproducible set of Fortran results to compare against.
- [x] Write `validation/build_fortran_ref.sh`. It builds the Fortran with gfortran and no makedepend,
      using the same cpp flags as the Julia `cppdefs.jl`: `periodic_ew` on; `gotm_call`, `implicit`,
      `allow_particle` and `relaxation` off; netcdf off (Fortran netcdf isn't installed and isn't needed).
- [ ] Run single-threaded only, because the OpenMP `linerelax` is non-deterministic (threads race on
      shared data).
- [ ] Use the full-state dump the Fortran already has (`w_pickup`: h,u,v,w,uf,vf,wf,T,S,Tr,p,gradhn,
      hxn,hyn as raw stream binary). Setting `pickup_int=1` gives the full state at every step.
- [ ] Test cases:
  1. `model` default, which starts at rest with stratification. Sanity check: it should stay at rest.
  2. `wiggle`, a wiggling front with real dynamics. This is the main test.
  3. Later: `wiggle` with `use_Shchepetkin: true` and `fnhhy=0` (hydrostatic), to cover the other branches.
- [ ] Also record the per-step kinetic energy and the solver residuals and iteration counts that the
      Fortran prints.

## Phase 2: Julia comparison harness (done; see CHANGES.md)
- [ ] Add a Julia `w_pickup_bin` that writes the same raw format as the Fortran (the current Julia
      pickup writes NetCDF).
- [ ] Write `validation/compare.jl`. Per step and per field it reports the max absolute difference,
      the max relative difference (normalised by the field's max |value|) and the location of the
      largest difference.
- [ ] Write `validation/run_case.sh <case>`, which assembles the experiment, runs Fortran and Julia,
      and compares the results.
- [ ] **Routine-level differential testing**, for when the whole model diverges and the cause isn't
      obvious. This revives the approach the original porter used: build the Fortran as a shared
      library (`PSOM_LIB.so`), set the Fortran `header` module globals from Julia (gfortran symbol
      names are `__header_MOD_<name>`), call one Fortran routine and its Julia twin on identical
      inputs, and compare the outputs. This narrows each bug down to a single routine.
- Pass criteria:
  - Step 1 should agree to ~1e-12 relative difference, apart from where Fortran uses single
    precision (see Phase 3).
  - After 10–100 steps, differences should grow only at round-off rate. The hydrostatic case should
    be close to bitwise identical.

## Phase 3: Fix translation bugs (done for wiggle and waves; see CHANGES.md)
Order: find the first routine in the timestep where results diverge, fix it, re-run, repeat. The
bugs already found by reading the code are below, roughly in the order they are hit within a step:

| # | File | Bug | Fortran |
|---|------|-----|---------|
| 1 | `advection_and_mixing.jl:160` | tracer update nested inside `selvar==5`, so tracers never advance | outside |
| 2 | `uvchy.jl:36` | `wy[i+1,j+1,k+1]` | `wy(i,j,k)` |
| 3 | `intpol_3rdorder.jl:47` | `cy[i0+1,j+1,k+1]` | `cy(i0,j,k)` |
| 4 | `rpevalgrad_Song.jl:48,97,100,181` | `zf` k-offset, `Jy2`/`dzsig` level shift, `rho`/`zc` j-offset | see file |
| 5 | `rpevalgrad_Sche.jl:177` | `rx[NI-1]` | `rx(NI)` |
| 6 | `srcface_3rdorder.jl:122` | `bbj[i,j+1]` | `bbj(i,j)` |
| 7 | `calcskfc.jl:14` | `ux[i+1,j+1]` | `ux(i,j)` |
| 8 | `hfill.jl:22`, `hsolve.jl` | north boundary uses the south sum (`vfbcs`) | `vfbcn` |
| 9 | `mprove.jl:88-124` | residual is missing `rhs -`; out-of-scope `i,j` in the error message | |
| 10 | `cpfine.jl:12-22` | `edt` multiplies the source terms, `sifc`/`sjfc` signs wrong, `gqj3` indices shifted | |
| 11 | `pbc.jl:84` | `fn -= wfbcb` | `fn - edt*wfbcb` |
| 12 | `sor.jl:6-7` | `for i in nxm`; `cp[2]` neighbour wrong | `do i=1,nxm`; `p(i+1,j,k)` |
| 13 | `dgtsl.jl:20-22` | pivot swaps the global `p` instead of `b` | |
| 14 | `prolong.jl:12,23` | `x+5e0`; `println[...]` | `x+.5` |
| 15 | `mgrid.jl:149-157` | `rhs_r` used before it is defined; `p_base` has lost its offsets | |
| 16 | `conadjust_sT.jl:22-23` | tracer indices `k`/`j+1` (fixed 2026-10-18, `18932c1`) | `k+1` |
| 17 | `cdiv.jl:10` | `wz[i,j,k]` | `wz(i,j,k+1)` (diagnostic only) |
| 18 | `diag_n2budget.jl:60,81` | `avezf[k-1]` at k=1 is out of bounds | Fortran reads out of bounds silently |
| 19 | `main.jl` | stray `then`; initial output written even on pickup; `save_parti(step)` signature | |
| 20 | `wiggle/src/linerelax_periodicew.jl` | `Random.seed!` on every call | `ran3` sequence keeps advancing |

Two known places where Julia and Fortran will legitimately differ:
- **Fortran single-precision locals** (`rpevalgrad_Song` const/Jsum, Sche FC/rx/adim). For the
  comparison, temporarily make the Julia code use `Float32` in exactly those places to confirm the
  match. Afterwards, keep Float64 as the documented choice.
- **Fortran `linerelax` out-of-bounds indexing on coarse grid levels** (`istart=NI/2`). This is a
  Fortran bug, and Julia's `mod` behaviour is the correct one. Tolerances there will be looser, or I
  will build a patched Fortran just for that check.

`waves` (`sponge` is never included, plus other errors) is fixed after `wiggle` passes.
`gotm_call`, `implicit` and particles stay out of scope until the core model is validated.

## Phase 4: Clean-up and CPU performance (largely superseded by Phase 5: the kernel port removed the allocations, slice copies and threading issues)
- [ ] Turn the code into a package with a `Project.toml`/`Manifest.toml`, and remove the runtime `Pkg.add`.
- [ ] Remove the dead `@ccall` comments and `fortVar.jl` stubs (once the differential tests no longer
      need them).
- [~] Preallocate all scratch arrays once: the mgrid `cp`/`rhs`/`res` arrays (~410 MB per call for
      waves) are done; advection, rpevalgrad, srcface, hsolve and mprove are still to do.
- [ ] Remove per-column and per-boundary slice copies (`linerelax`, `hbc`, `hfill`), and the
      O(NI²) `diag_n2`.
- [ ] Store each multigrid level as its own array instead of `reshape(view(vector))`.
- [ ] Move `exit()` out of the numerical routines; check an error flag afterwards instead.
- [ ] Make the threaded chunking safe for any thread count.
- [x] Benchmark CPU time per step against Fortran (`validation/bench.sh`; baseline in CHANGES.md).

## Phase 5: GPU port (done, apart from running on a real GPU; see CHANGES.md section 6, GPU_PORTING.md, and TODO.md for what is left)
Design, as implemented:
- [x] One source for CPU and GPU with KernelAbstractions; backend chosen by `PSOM_BACKEND`
      (cpu | cuda | amdgpu, plus jlarray and opencl for testing without a GPU).
- [x] Instead of a `State` struct: the header arrays stay host arrays (init, I/O, diagnostics unchanged).
      `DEV` is a NamedTuple of their device copies (the arrays themselves on the CPU), passed into kernels
      as arguments. OffsetArrays of GPU arrays work inside kernels, so index offsets are kept.
- [x] Every routine of the time step as kernels (pointwise, column, reductions, colour sweeps).
- [x] Solvers: multi-colour line Gauss–Seidel (`linerelax`) and SOR (`hsolve`/`mprove`), also in the
      Fortran (D3). The sequential coarsest-level `sor` and the boundary fills (`mgpfill`, `hfill`) are
      kept exact as wavefronts / single-work-item sweeps.
- [x] Host/device traffic: state on the device; copies to the host only on output steps.
- [x] Checks: CPU backend bit-identical to the pre-port code (1 and 4 threads); emulated GPU bit-identical;
      GPU-compiled kernels (OpenCL/SPIR-V) at round-off; routine-level test against the Fortran passes.
- [ ] Run on the GPU server: `validation/backend_check.sh cuda`, `validation/gpu_bench.sh cuda`.
- [ ] Profile on the GPU, then decide on the solver changes that alter results at tolerance level
      (free-surface solver convergence checks or CG; parallel boundary fills; fewer launches).
- [ ] Only if the GPU turns out to be a consumer card (slow FP64): a mixed-precision variant.

## Target GPU server: what's needed to decide (the remote server is not known yet)
Using KernelAbstractions keeps the code vendor-neutral, so Phases 1–4 and most of Phase 5 don't depend
on the server. These are the details that decide the remaining choices:

| Need | Why it matters |
|------|----------------|
| GPU vendor and model (e.g. NVIDIA A100/H100/L40S/RTX 4090, AMD MI250/MI300) | Picks the backend (CUDA.jl / AMDGPU.jl / oneAPI.jl). |
| **FP64 throughput** (follows from the model) | PSOM runs in Float64. Data-centre GPUs (A100, H100, MI250) are fast in FP64. Consumer and workstation cards (RTX, L40S, A6000) are 32–64× slower in FP64, which would force a mixed-precision redesign. **This is the biggest decision.** |
| GPU memory per card, and number of GPUs per node | Decides whether the largest target grid fits on one GPU or needs multi-GPU domain decomposition (MPI.jl with halo exchange), which is a large extra phase. |
| Driver / CUDA (or ROCm) version, OS | Compatibility with current CUDA.jl/AMDGPU.jl. |
| Can Julia be installed in user space, and does the node have internet access (for packages)? Which scheduler (SLURM?) | Deployment; an offline install needs a pre-built depot. |
| Grid sizes and run lengths you actually want | Decides whether a GPU is worth it (≥ ~10⁶ cells) and how much effort goes into the solvers. |

## Open questions for you
1. The GPU server details above, when you have them. I can do a lot of the work before then.
2. Which experiments matter most to you after `wiggle`, so I prioritise their overrides
   (waves, NA, shelfbreak, ...)?
3. (answered: commit per fix on the branch)
4. (answered: Fortran bugs fixed in both V1.0 and Julia_Port; see CHANGES.md, "Fortran bug fixes")
5. (answered: smooth sponge taper; `mldepth` from the namelist, default 10 m; multi-colour solvers in
   Fortran too. See CHANGES.md, "Decisions implemented")
