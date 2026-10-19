# TODO: finishing the GPU port

State at the end of October 2026 (branch `fix-and-validate`): the whole time step of the Julia model runs as
KernelAbstractions kernels and has been validated on the CPU, on an emulated GPU array and with the GPU
compiler (OpenCL, run on a CPU). It has **not yet run on a real GPU**. Background: `GPU_PORTING.md` (design,
rules, server instructions), `VALIDATION.md` (tests), `CHANGES.md` (what was done).

Items marked **[changes results]** alter the solution at the solver-tolerance level. For those, check the
Fortran comparison against the round-off tolerances in `VALIDATION.md` (or agree on new ones), then save a new
regression baseline (`validation/regress.sh save`).

## 1. First runs on the GPU server (needs access to the server)

- [ ] Record the server details in `PLAN.md` ("Target GPU server"): GPU model and **FP64 throughput**,
      memory per GPU, GPUs per node, driver and CUDA/ROCm version, scheduler, internet access on compute nodes.
- [ ] Install Julia and the GPU package, and check that the GPU is visible (`GPU_PORTING.md`, "Running on a GPU
      server", steps 1–2).
- [ ] Correctness: `validation/backend_check.sh cuda` (or `amdgpu`). Expected: round-off differences, ≲1e-10
      relative, with `hxn` in waves ~1e-7 (a noise-only field). Anything larger is a bug: find the first routine
      that differs by tracing the device arrays after every routine (that's how the `diag_n2budget` side effects
      were found; see `CHANGES.md`, section 6).
- [ ] Speed: `validation/gpu_bench.sh cuda 10`, plus a CPU reference on the same node
      (`validation/gpu_bench.sh cpu 10 <threads>`). Put the target grid sizes into `SIZES`. Record the results
      in `CHANGES.md`.
- [ ] Profile one representative grid (Nsight Systems: `nsys profile julia ...`, or `CUDA.@profile`) to see
      the time per kernel, the launch overhead and the host–device synchronisations.

## 2. Performance (decide from the profile)

- [ ] **Free-surface solver** [changes results]. `hsolve` never reaches its tolerance (1e-12) and always does
      all 100 SOR iterations; `mprove` then does another 114–433. Every iteration does a reduction for the
      residual with a host sync, and each `hsolve` iteration calls the sequential `hfill` three times.
      Options, cheapest first:
      - check the residual only every N iterations (fewer syncs);
      - a tolerance that can actually be reached, so the iterations stop early;
      - a Krylov solver (conjugate gradient if the 9-point matrix `ch` is symmetric, else BiCGSTAB), with a
        simple preconditioner.
- [ ] **Sequential boundary sweeps** [changes results]. These are kept exactly as in the Fortran, which needs
      sequential work on the GPU: the `hfill` ghost rows (one work item per row), the `mgpfill` faces
      (single-workgroup wavefronts), and the coarsest-level `sor` (single-workgroup wavefront). Replace them
      with parallel (Jacobi-type or red–black) updates if the profile shows they matter.
- [ ] **Small kernel launches.** Fuse kernels that are launched back to back over the same range (e.g.
      `efill`: 8 launches per call; the colour sweeps), and try other workgroup sizes.
- [ ] **Host syncs in convergence tests** (`res9_max`, the multigrid residual): keep the decision on the device
      where possible.
- [ ] **Mixed precision**, only if the GPU is a consumer or workstation card with slow FP64 (RTX, L40S, A6000):
      then decide which parts can run in Float32. Not needed on A100/H100/MI250-class cards.

## 3. Functionality not yet on the GPU

- [ ] **Diagnostics.** `diag_n2` and `diag_n2budget` run on the host, on output steps only, after copying the
      state back. As a result, the `rhoprev` time derivative in `diag_n2budget` covers the time since the last
      output instead of one step. Decide whether that's acceptable, or port these diagnostics to kernels.
- [ ] **Particles** (`allow_particle`), **relaxation**, **gotm** and the **implicit** vertical mixing are
      compiled out in the tested experiments and have not been ported to kernels (or validated). With particles
      on, they would need the velocity field on the host every step. Port whichever the target experiments
      need.
- [ ] **Other experiments** (NA, shelfbreak, NP_summer, NP_winter, BoB, leewaves, freewaves, …): their
      experiment-specific Julia routines haven't been compared with the Fortran or ported to kernels. Only
      `model`, `wiggle` and `waves` have. For each experiment needed: run `validation/run_case.sh` and
      `difftest.jl`, fix differences, then port its routines following `GPU_PORTING.md` and check with
      `validation/check_kernels.jl <exp>`.
- [ ] **Grids larger than one GPU** (memory is ~1 kB per cell, so ~70M cells on an 80 GB card): this would need
      domain decomposition across GPUs (MPI.jl with halo exchanges), which is a large job. Only if the target
      grids need it.

## 4. Validation still to do

- [ ] Full-size waves (16×768×192) against the Fortran. Its static arrays exceed 2 GB, which can't be linked on
      macOS. On Linux, build with `-mcmodel=medium` (see GET_STARTED_3, CASE 4).
- [ ] Longer runs (≥1000 steps) against the Fortran, to check that differences stay at round-off growth.
- [ ] After any change marked [changes results]: the four regression cases against the Fortran, at the
      tolerances in `VALIDATION.md`.

## 5. Code clean-up

- [ ] Give the model its own `Project.toml`/`Manifest.toml` (today it borrows `validation/`) and remove the
      runtime `Pkg.add` in `code/model/inc/includeFile.jl`.
- [ ] Remove the leftovers of the original hybrid Julia/Fortran port: the commented `@ccall "./PSOM_LIB.so"`
      lines and the `fortVar.jl` stubs (`@fortGetArr`).
- [ ] Replace `exit(1)` inside numerical routines (`hsolve`, `mprove`, `linerelax`, `meanh`, `checks`) with
      an error flag checked after the step, so a GPU run can stop cleanly.
- [ ] Continuous integration: run `check_kernels.jl`, `regress.sh` and `backend_check.sh jlarray` on GitHub
      Actions for every push (CPU-only runners are enough).
