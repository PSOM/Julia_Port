# CPU timing of the Julia model: time per step after warm-up (excludes compilation and init).
#
# Usage: [PSOM_SIZE="NI NJ NK ngrid"] [PSOM_BACKEND=cpu|cuda|amdgpu] julia -t<threads> --project=validation validation/bench.jl <experiment> [nsteps=10]
# Needs runs/<exp>[_size]/julia/namelist.yml (written by run_case.sh). Output and pickups are disabled.

const expe = length(ARGS) >= 1 ? ARGS[1] : "wiggle"
const nbench = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 10

const VAL = @__DIR__
const CODE = joinpath(VAL, "..", "code")
const SIZE_SFX = haskey(ENV, "PSOM_SIZE") ? "_" * join(split(ENV["PSOM_SIZE"])[1:3], "x") * "g" * split(ENV["PSOM_SIZE"])[4] : ""
const EXE = joinpath(CODE, "exe", expe == "model" ? "nh" : "nh_" * expe)
const RDIR = joinpath(VAL, "runs", expe * SIZE_SFX, "bench")

run(`sh $(joinpath(CODE, "tools", "compile_julia.sh")) $expe`)
haskey(ENV, "PSOM_SIZE") && run(`bash -c "source $(joinpath(VAL, "size_override.sh")); write_size_jl $(joinpath(EXE, "inc", "size.jl"))"`)
# no file output: time the computation only
let f = joinpath(EXE, "inc", "cppdefs.jl")
    s = read(f, String)
    s = replace(s, r"(const +file_output *=).*" => s"\1false")
    write(f, s)
end
mkpath(RDIR)
let src = joinpath(VAL, "runs", expe * SIZE_SFX, "julia", "namelist.yml")
    s = read(src, String)
    s = replace(s, r"pickup_int:.*" => "pickup_int: 1000000", r"dirout:.*" => "dirout: \"$RDIR/\"",
                r"nsteps:.*" => "nsteps: $(nbench + 1)")
    write(joinpath(RDIR, "namelist.yml"), s)
end
cd(RDIR)
empty!(ARGS); push!(ARGS, joinpath(RDIR, "namelist.yml"))
include(joinpath(EXE, "src", "main.jl"))

using Printf, Statistics, Profile
redirect_stdout(devnull) do
    global pcorr, istep = initialize()
end
t1 = @elapsed (redirect_stdout(() -> step!(pcorr, istep + 1), devnull); dsync())   # includes compilation
times = Float64[]; allocs = Int[]; bytes = Int[]
for st in istep+2:istep+1+nbench
    r = @timed (redirect_stdout(() -> step!(pcorr, st), devnull); dsync())
    push!(times, r.time); push!(bytes, r.bytes)
end
@printf("julia %s%s backend=%s threads=%d  first step (with compilation) %.2f s\n", expe, SIZE_SFX, PSOM_BACKEND, Threads.nthreads(), t1)
@printf("  per step: median %.4f s  min %.4f s  (%d steps)  allocated %.1f MB/step\n",
        median(times), minimum(times), nbench, mean(bytes) / 2^20)

# PSOM_PROFILE=1: sampled profile of 3 more steps, flat report of the model's own functions
if get(ENV, "PSOM_PROFILE", "0") == "1"
    Profile.clear()
    @profile redirect_stdout(() -> for st in 1:3; step!(pcorr, istep + 1 + nbench + st); end, devnull)
    open(joinpath(RDIR, "profile.txt"), "w") do io
        Profile.print(IOContext(io, :displaysize => (200, 250)); format=:flat, sortedby=:count, mincount=20)
    end
    println("profile written to ", joinpath(RDIR, "profile.txt"))
end
