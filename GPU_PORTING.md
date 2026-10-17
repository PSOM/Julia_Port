# Porting PSOM routines to GPU kernels: conventions

The Julia model runs on the CPU or on a GPU from one source, using
[KernelAbstractions.jl](https://github.com/JuliaGPU/KernelAbstractions.jl) (KA). The machinery lives in
`code/model/src/arch.jl`. Read that file first: it is short.

## The model state: HOST and DEV

- The header arrays (`header.jl`: `u`, `v`, `h`, `cx`, `Jac`, …) are ordinary host arrays. Initialisation, I/O
  and most diagnostics use them directly, as before.
- `HOST` is a `NamedTuple` of all header arrays (`HOST.u === u`). `DEV` is the same set on the
  compute device, with the same names and index offsets (`DEV.u[i,j,k,n]` is `u[i,j,k,n]`).
  - On the CPU backend (the default), `DEV === HOST`: no copies. So ported and not-yet-ported routines can be
    mixed, and the results must stay **bit-identical** to the pre-port code.
  - On a GPU, `DEV` holds device copies. `to_device!()` runs at the end of `initialize()`, and `to_host!()`
    runs before output.
- Backend: environment variable `PSOM_BACKEND` = `cpu` (default) | `jlarray` | `cuda` | `amdgpu`.
  `jlarray` (JLArrays.jl) emulates a GPU array on the CPU and **forbids scalar indexing**, so it tests the
  GPU code path without a GPU.

## How a routine is ported

1. Every loop nest becomes a `@kernel` that does the work of **one** loop iteration. It is launched over the
   loop's index ranges with `launch!`:
   ```julia
   @kernel function k_coriolis!(si, sj, sk, ffc, bbc, u, v, w, drpx, drpy, uvis, vvis, wvis,
                                n, fac, fac2, ainv, betainv, fnhhy, lo)
     i, j, k = @index(Global, NTuple) .+ lo      # `lo` is always the LAST kernel argument
     @inbounds begin
       si[i, j, k] = -ffc[i, j] * v[i, j, k, n] + ...
     end
   end

   function coriolis(n, S=DEV)                    # array set S is the LAST argument, default DEV
     local fac = EPS * delta
     ...
     launch!(S, k_coriolis!, (1:NI, 1:NJ, 1:NK), S.si, S.sj, ..., n, fac, ..., fnhhy)
   end
   ```
   The ranges tuple is in **array index order (i, j, k)**, so i is the fastest-varying index, which matches
   the memory layout. `launch!` passes `lo` (= first index − 1 of each range) as the last argument.
2. **Everything a kernel uses must be an argument.** That covers arrays (`S.name`, or a work array) and the
   values of **non-const globals** (`dtf`, `qpr`, `fnhhy`, `kappah`, `lambda`, `delta`, `beta`,
   `use_shchepetkin`, …, i.e. anything declared `global x::T` in `header.jl` or set in `ini_param.jl`).
   - Compile-time constants (`const` scalars such as `NI`, `NJ`, `NK`, `EPS`, `PI`, `rc_kind`, and the
     `cppdefs` flags) may be used directly.
   - **Never use a header array by its global name inside a kernel.** It works on the CPU but not on a GPU.
     `validation/check_kernels.jl` checks this.
3. **Kernels must be data-parallel.** No work item may read a value that another work item of the same
   launch writes.
   - Loops with a dependency along one index (vertical integrals, tridiagonal solves, running sums) become
     one work item per column (or row), with the dependent loop *inside* the kernel:
     `launch!(S, k_x!, (1:NI, 1:NJ), …)` with `for k in 1:NK … end` in the kernel.
   - A loop that writes `a[i]` and then reads `a[i-1]` in the same sweep is a dependency; restructure it.
   - Gauss–Seidel / SOR sweeps use the multi-colour ordering already implemented in `linerelax`, `hsolve`
     and `mprove` (`ncolours_i`, `colour_irange`). Use one launch per colour.
4. **Bit-identical on the CPU**: keep each floating-point expression in the same order as the existing
   Julia code (which matches the Fortran). In particular, keep sums sequential in the original order: a sum
   over k stays a loop over k inside a column kernel, never `sum(...)` or a parallel reduction. Max and min
   reductions are order-independent and may use `maximum(A)` on a work array.
5. **Reductions**: write the per-point values into a work array with a kernel, then `maximum(abs, view)` or
   similar. These Base reductions work on device arrays.
6. **Work arrays** (function-local temporaries): `tmp = work(S, :unique_name, rc_kind, dims...)`, where
   dims are integers or ranges (ranges give an OffsetArray). It is allocated once per backend and name. Its
   contents are *not* zeroed on reuse: initialise what you read. Use a name prefixed with the routine name
   unless the array is genuinely shared.
7. **No I/O, `println`, `exit`, `error`, allocation or dynamic dispatch inside kernels.** For error checks
   (e.g. `dgtsl` returning `info != 0`), write a flag into a work array, check it on the host after the
   launch, and print or exit there.
8. Whole-array assignments: `dfill!(A, x)`. On index ranges: `dfill!(S, A, x, ranges...)`,
   `dcopy!(S, A, B, ranges...)`, `daddto!(S, A, B, ranges...)`. Do not use broadcasting (`@. a[1:NI,…] = …`)
   on device arrays: views of `OffsetArray`s of GPU arrays can fall back to scalar indexing.
9. Callers:
   - Calls inside the time step (`momentum`, `step!`) use the default `DEV`, and array arguments come from
     `DEV` (e.g. `hsolve(DEV.h, DEV.oldh, DEV.hdt, dtim)`).
   - Calls during `initialize()`, and calls from routines that are **not yet ported** (they work on the host
     globals), pass `HOST` explicitly (e.g. `evalrho(rho, n, HOST)` inside an unported `rpevalgrad_Song`).
     When you port a caller, switch its calls to pass its own `S`.
10. Keep the comments that explain the physics or the Fortran correspondence. Remove the old commented-out
    `@ccall` lines in the files you port.
11. Experiment overrides (`code/<exp>/src/*.jl`, e.g. `waves`) are separate copies: port them the same way
    when you port the model version.

## Checking

- `validation/regress.sh check [wiggle sche hydro waves]` reruns the Julia model (CPU backend) and compares
  every step's full state with the saved pre-port baseline (expected **0.000e+00**) and with the Fortran
  reference (expected round-off, ~1e-10 or less; `hxn` in waves ~1e-7 is noise, see CHANGES.md). Takes
  about 6 minutes for all four cases.
- `PSOM_BACKEND=jlarray validation/regress.sh check wiggle` runs on the emulated GPU array. This only works
  once every routine called in the time step is ported.
- `julia validation/check_kernels.jl` lists kernel bodies that use globals they must not use (see 2.). It does
  not look inside ordinary functions called from kernels (such as `potdens`, `dgtsl`): those must also use
  only their arguments and constants.
