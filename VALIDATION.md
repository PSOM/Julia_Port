# Validation of the Julia port against the Fortran

How the Julia port was checked against the Fortran PSOM, how to rerun the checks, and the latest results.
What was changed is in `CHANGES.md`. All tools are in `validation/`, which has its own Julia project
(`validation/Project.toml`). Build products (`validation/build/`) and run output (`validation/runs/`) are
not tracked.

The **Fortran is the reference**: the Fortran sources in `code/` (the same as `PSOM/V1.0`, branch
`fix-fortran-bugs`), built with gfortran. The tests are:

| # | Test | What it shows | Tool |
|---|------|---------------|------|
| 1 | Whole model, Fortran vs Julia | The two codes give the same solution, step by step, field by field | `run_case.sh`, `compare.jl` |
| 2 | Routine by routine, Fortran vs Julia | Each routine gives the same output on identical input; pinpoints any difference to one routine | `difftest.jl` |
| 3 | Julia regression baselines | A code change (e.g. the GPU port) leaves the results bit-identical | `regress.sh` |
| 4 | GPU backends vs CPU | The kernels give the same results on a GPU (round-off) | `backend_check.sh` |
| 5 | Kernel rules | No kernel uses host-only globals (would fail on a GPU) | `check_kernels.jl` |
| 6 | Fortran memory and threading checks | No uninitialised reads; OpenMP gives the serial answer | `build_fortran_ref.sh` variants |

## Test cases

| Case | Grid | Steps | What it exercises |
|------|------|------:|-------------------|
| `wiggle` | 24×24×24, 4 multigrid levels | 10 (also 100) | A wiggling front over flat topography: non-hydrostatic, Song pressure gradient, tracer, line relaxation with a random line start (`ran3`) |
| `wiggle`, `use_Shchepetkin=true` | same | 10–30 | Shchepetkin pressure-gradient scheme |
| `wiggle`, `fnhhy=0` | same | 10–30 | Hydrostatic mode (no non-hydrostatic pressure solve) |
| `waves` | 16×192×96, 4 levels (the native grid is 16×768×192) | 3 | Internal-wave experiment: wind stress, heat flux, sponges, N² diagnostics |

`waves` runs on a reduced grid because its static Fortran arrays at full size exceed 2 GB, which can't be
linked on macOS. `PSOM_SIZE="NI NJ NK ngrid"` (`size_override.sh`) builds both codes at another size; the
multigrid storage sizes are derived as in `mgrid`.

## 1. Whole model: `run_case.sh` and `compare.jl`

```
validation/run_case.sh wiggle 10                              # default wiggle
validation/run_case.sh wiggle 30 --set use_Shchepetkin=true --tag sche
validation/run_case.sh wiggle 30 --set fnhhy=0e0 --tag hydro
PSOM_SIZE="16 192 96 4" validation/run_case.sh waves 3
```

- Builds the Fortran reference (`build_fortran_ref.sh`: `code/model` + the experiment's files, as
  `tools/compile.sh` does; gfortran `-O2 -fdefault-real-8 -fdefault-double-8`; serial; the cpp flags are
  generated from the Julia `cppdefs.jl`, so both codes compile the same branches).
- Runs both codes from the same namelist (`--set key=value` changes both), writing the **full model
  state every step** in the Fortran pickup layout: h, u, v, w, uf, vf, wf, T, S, Tr, p, gradhn, hxn,
  hyn.
- `compare.jl` reports, per step and per field, the largest absolute difference, the **relative
  difference** (largest |Julia − Fortran| divided by the largest |Fortran| of that field), where it
  occurs, and the same over interior points only. `ke.txt` compares the printed kinetic energy.

**Pass criterion: round-off.** Two correct codes that do the same arithmetic in a slightly different order
differ by ~1e-16 per operation, and the difference grows slowly with the number of steps. The noise floor
measured on the Fortran itself (`-O2` vs `-O0`, 10 wiggle steps) is ≤4e-11. A translation error shows up
instead as a difference of 1e-8 or more, usually from step 0 or 1.

**Results** (current code, October 2026):

| Case | Steps | Kinetic energy, rel. diff (worst step) | Worst field, rel. diff |
|------|------:|------:|------:|
| wiggle | 10 | 3.9e-15 | 4.2e-11 (`hxn`) |
| wiggle | 100 | 6.6e-15 | 1.2e-10 (`wf`) |
| wiggle, Shchepetkin | 10 | 4.1e-15 | 4.7e-11 (`hxn`) |
| wiggle, hydrostatic | 10 | 4.7e-15 | 6.1e-11 (`hxn`) |
| waves 16×192×96 | 3 | 7.7e-14 | 4.2e-13 (`hyn`) in fields with real signal [1] |

[1] In waves, `w`, `wf` and `hxn` have relative differences of 6.3e-10, 6.4e-10 and 8.0e-8. These fields are themselves
pure round-off noise in this flow (it initially varies only in y, so x-gradients like `hxn` are ~1e-9 against
~5e-5 for `hyn`). Their absolute differences (~2e-17) are the same as for the other fields.

## 2. Routine by routine: `difftest.jl`

```
julia --project=validation validation/difftest.jl wiggle 1 synced
PSOM_SIZE="16 192 96 4" julia --project=validation validation/difftest.jl waves 2 synced
```

- The Fortran is built as a shared library (`libpsom_ref`) and loaded into the Julia process.
- Every Fortran `header` module variable that has a Julia global of the same name is mapped onto it (186
  variables for wiggle; types and sizes are checked).
- The model sequence (initialisation, then each time step with `momentum` split into its ~40 sub-steps)
  runs on both sides, and **all mapped variables are compared after every sub-step**.
- `synced` mode: the Julia state is copied into the Fortran before each sub-step, so each routine gets
  identical inputs. A difference therefore points to that one routine. `free` mode lets both run
  independently, to see where they drift apart.
- Extra unit tests on the current state: `chfine`, `hfill` and `mprove` (inside the free-surface solver),
  and `conadjust` on a state made statically unstable (none of the short test runs convect, so the
  regular sequence never exercises it: 2091 cells convect in the unit test).
- Pass: relative difference ≤1e-12 (default `tol`) for every variable after every sub-step.

This is the tool that found nearly all the translation bugs listed in `CHANGES.md`. **Current result**:
every sub-step "ok" for wiggle; the only line above 1e-12 is the `pv` diagnostic at initialisation
(2.5e-12, round-off).

## 3. Julia regression baselines: `regress.sh`

```
validation/regress.sh save                                    # store the current results as the baseline
REGRESS_BASELINE=julia_baseline_hbc validation/regress.sh     # rerun all four cases and compare
```

Runs the four test cases and compares every field at every step with a saved Julia baseline (expected:
**bit-identical**, difference exactly 0) and with the Fortran. Used for every step of the GPU port: each
ported routine had to leave all four cases bit-identical. The baseline in use is `julia_baseline_hbc`,
saved after the one deliberate arithmetic change of the port (sequential sums in `hbc`, ≤8e-11).
`PSOM_THREADS=4` runs Julia with 4 threads (also bit-identical); `PSOM_BACKEND` selects the backend.

## 4. GPU backends: `backend_check.sh`

```
validation/backend_check.sh cuda        # or amdgpu; jlarray and opencl run without a GPU
```

Runs each case on the CPU backend and on the given backend and compares all fields at every step.

| Backend | What it is | Result |
|---------|------------|--------|
| `jlarray` | GPU array emulated on the CPU; forbids element-wise access from the host, like a real GPU | bit-identical, all four cases |
| `opencl` | kernels compiled by the GPU compiler (GPUCompiler → SPIR-V), run by the pocl CPU driver | round-off: wiggle 4.5e-11, Shchepetkin 4.0e-11, hydrostatic 7.4e-11; waves ≤5.5e-10 (`w`) |
| `cuda`, `amdgpu` | real GPUs | not run yet (no GPU available); see `GPU_PORTING.md` |

Round-off differences are expected on a GPU, because GPU compilers fuse multiplies and adds.

## 5. Kernel rules: `check_kernels.jl`

`julia validation/check_kernels.jl` scans every kernel for header arrays or non-constant globals used by
name (these work on the CPU but don't exist on a GPU), and every kernel launch for host arrays passed
instead of their device copies. Result: OK.

## 6. Fortran memory and threading checks

`build_fortran_ref.sh` options: `FFLAGS=...`, `BUILD_SUFFIX=_name`, `OMP=1` (OpenMP build),
`PSOM_CODE=/path/to/V1.0/code` (build another source tree).

- **Uninitialised memory:** `FFLAGS="... -frecursive -finit-real=snan -ffpe-trap=invalid"`. Every local is
  on the stack and starts as a signalling NaN, which stops the program when used. This found the
  `staticsigma` bug (F5 in `CHANGES.md`). After the fixes it runs clean for wiggle (all three
  configurations) and waves. Builds with `-frecursive` are bit-identical to normal builds.
- **OpenMP:** `OMP=1`, run with `OMP_NUM_THREADS=1, 3, 4`. After the multi-colour solvers (D3), every
  thread count is bit-identical to the serial build, and repeated runs are identical (wiggle 10 steps,
  waves 3 steps). The OpenMP build needs a bigger main-thread stack; see `CHANGES.md`, "Stack size".
- **Fortran tree vs Fortran tree:** `PSOM_CODE=.../V1.0/code` gave bit-identical results to the copies in
  `Julia_Port/code` after the Fortran fixes (wiggle, waves).

## Speed: `bench.sh`, `bench.jl`, `gpu_bench.sh`

`validation/bench.sh <exp> [nsteps] [threads]` times Fortran and Julia per step (after warm-up, no output);
`gpu_bench.sh <backend>` sweeps grid sizes from 0.3M to 19M cells. Results so far are in `CHANGES.md`.

## Reproducing the history of the fixes

`CHANGES.md` has a table of the Fortran–Julia agreement after each translation fix. It was produced by
checking out each fix commit into its own git worktree and running test 1 against the original, unpatched
Fortran (from commit `aa2085a`, before any Fortran change).
