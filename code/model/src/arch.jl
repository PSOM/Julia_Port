# Backend (CPU / GPU) selection and the device copy of the model state.
#
# The header arrays (header.jl) stay ordinary host arrays: initialisation, I/O and diagnostics use
# them as before. Ported routines run KernelAbstractions kernels on an "array set" S, a NamedTuple
# with one array for every header array (same names, same index offsets: S.u[i,j,k,n] is u[i,j,k,n]):
#   HOST = the host header arrays themselves,
#   DEV = their device copies (on the CPU backend DEV's arrays ARE the header arrays: no copies, no
#       synchronisation, so ported and not-yet-ported routines can be mixed).
# Ported routines take S as their last argument, default DEV (time stepping); initialize() passes HOST,
# because it runs before the state is copied to the device. to_device!() copies HOST -> DEV at the end of
# initialize(), to_host!() copies DEV -> HOST before output and host-side diagnostics.
#
# Select the backend with the environment variable PSOM_BACKEND:
#   cpu (default) | cuda (needs CUDA.jl) | amdgpu (needs AMDGPU.jl) |
#   jlarray (JLArrays.jl: a GPU array emulated on the CPU; no scalar indexing allowed, so it checks
#            the GPU code path on a machine without a GPU) |
#   opencl  (OpenCL.jl with the pocl CPU driver: kernels are compiled by the GPU compiler stack
#            (GPUCompiler -> SPIR-V) and run on the CPU, so it also checks that every kernel compiles
#            for a GPU: no dynamic dispatch, allocation or exceptions inside kernels)

using KernelAbstractions
const KA = KernelAbstractions
using Adapt

const PSOM_BACKEND = get(ENV, "PSOM_BACKEND", "cpu")
@static if PSOM_BACKEND == "cuda"
    using CUDA
    const backend = CUDABackend()
    const DevArray = CuArray
elseif PSOM_BACKEND == "amdgpu"
    using AMDGPU
    const backend = ROCBackend()
    const DevArray = ROCArray
elseif PSOM_BACKEND == "opencl"
    using pocl_jll, OpenCL
    const backend = OpenCLBackend()
    const DevArray = CLArray
elseif PSOM_BACKEND == "jlarray"
    using JLArrays
    const backend = JLBackend()
    const DevArray = JLArray
elseif PSOM_BACKEND == "cpu"
    const backend = KA.CPU()
    const DevArray = Array
else
    error("PSOM_BACKEND=$(PSOM_BACKEND): use cpu, cuda, amdgpu, opencl or jlarray")
end
const on_cpu = backend isa KA.CPU
println("PSOM backend: ", PSOM_BACKEND, " (", nameof(typeof(backend)), ")")

# device array with the same index offsets as the host array A (A itself on the CPU backend)
to_dev(A::AbstractArray) = on_cpu ? A : adapt(DevArray, A)
to_dev(A::OffsetArray) = on_cpu ? A : OffsetArray(adapt(DevArray, parent(A)), axes(A))

# zero-initialised device scratch array; dims are integers or ranges (ranges give an OffsetArray)
dzeros(T::Type, dims::Integer...) = KA.zeros(backend, T, dims...)
dzeros(T::Type, ranges::AbstractUnitRange...) = OffsetArray(KA.zeros(backend, T, length.(ranges)...), ranges...)

# The header arrays: every const global array of numbers defined so far (header.jl comes before
# this file).
const header_array_names = Tuple(sort([n for n in names(@__MODULE__; all=true)
    if isdefined(@__MODULE__, n) && isconst(@__MODULE__, n) &&
       (x = getfield(@__MODULE__, n); x isa AbstractArray && eltype(x) <: Number && !(x isa AbstractRange))]))

const HOST = NamedTuple{header_array_names}(map(n -> getfield(@__MODULE__, n), header_array_names))
const DEV = NamedTuple{header_array_names}(map(n -> to_dev(HOST[n]), header_array_names))

# backend that runs kernels on the array set S (on the CPU backend, typeof(DEV) == typeof(HOST))
backend_of(::typeof(HOST)) = KA.CPU()
backend_of(::typeof(DEV)) = backend

# Work arrays of the ported routines, allocated once per array set and name:
#     tmp = work(S, :name, T, ranges...)     (ranges as for dzeros; contents are not preserved
#                                           between different users of the same name)
const _work = Dict{Tuple{Bool,Symbol},Any}()
function work(S, name::Symbol, T::Type, dims...)
    get!(_work, (backend_of(S) isa KA.CPU, name)) do
        backend_of(S) isa KA.CPU ? (eltype(dims) <: Integer ? zeros(T, dims...) :
                                    OffsetArray(zeros(T, length.(dims)...), dims...)) : dzeros(T, dims...)
    end
end

_copy!(dst::OffsetArray, src::OffsetArray) = copyto!(parent(dst), parent(src))
_copy!(dst, src) = copyto!(dst, src)

"copy the host header arrays (all, or the given names) to the device; no-op on the CPU backend"
function to_device!(names=header_array_names)
    on_cpu && return
    for n in names
        _copy!(DEV[n], HOST[n])
    end
    dsync()
end

"copy the device arrays (all, or the given names) back to the host header arrays; no-op on the CPU backend"
function to_host!(names=header_array_names)
    on_cpu && return
    dsync()
    for n in names
        _copy!(HOST[n], DEV[n])
    end
end

# ---------------------------------------------------------------------------------------------
# Kernel launches. A loop nest  for k in rk, j in rj, i in ri  (i fastest) becomes
#     launch!(S, k_name!, (ri, rj, rk), args...)
# with a kernel whose last argument is `lo` (first index of each range minus 1) and that starts with
#     i, j, k = @index(Global, NTuple) .+ lo
# Kernels must be data-parallel: no iteration may read what another one writes.

@inline function launch!(S, kernel, ranges::Tuple, args...)
    lo = map(r -> first(r) - 1, ranges)
    n = map(length, ranges)
    any(==(0), n) && return
    kernel(backend_of(S))(args..., lo; ndrange=n)
    return
end

"wait for all kernels (needed before host-side use of device data, e.g. timing)"
dsync() = PSOM_BACKEND == "jlarray" ? nothing : KA.synchronize(backend)   # JLArrays runs synchronously

# ---------------------------------------------------------------------------------------------
# Generic helpers on index ranges (any number of dimensions); on whole arrays use dfill!(A, x).
@kernel function k_fill!(A, x, lo)
    I = @index(Global, NTuple) .+ lo
    @inbounds A[I...] = x
end
@kernel function k_copy!(A, B, lo)
    I = @index(Global, NTuple) .+ lo
    @inbounds A[I...] = B[I...]
end
@kernel function k_addto!(A, B, lo)
    I = @index(Global, NTuple) .+ lo
    @inbounds A[I...] += B[I...]
end
"A[ranges...] .= x"
dfill!(S, A, x, ranges...) = launch!(S, k_fill!, ranges, A, x)
"A[ranges...] .= B[ranges...]"
dcopy!(S, A, B, ranges...) = launch!(S, k_copy!, ranges, A, B)
"A[ranges...] .+= B[ranges...]"
daddto!(S, A, B, ranges...) = launch!(S, k_addto!, ranges, A, B)
"copy the whole array B into A (same shape and offsets)"
dcopy!(A::AbstractArray, B::AbstractArray) = (_copy!(A, B); nothing)
"fill the whole array A (all its indices, including halos)"
dfill!(A::AbstractArray, x) = (fill!(A isa OffsetArray ? parent(A) : A, x); nothing)
