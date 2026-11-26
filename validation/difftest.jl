# Routine-level differential test: Julia port vs Fortran reference, inside one process.
#
# The Fortran model is loaded as a shared library (build_fortran_ref.sh builds libpsom_ref).
# Every Fortran `header` module variable that has a Julia global of the same name (case-insensitive)
# is mapped onto it. The model sequence (init + time steps, with momentum split into its sub-steps)
# is then run on both sides, and after every sub-step all mapped variables are compared.
#
#   mode "synced" (default): before each sub-step the Julia state is copied into Fortran, so each
#                            routine is tested on identical inputs -> isolates bugs routine by routine.
#   mode "free":             both sides run independently -> shows where the solutions start to drift.
#
# Usage (from anywhere):
#   [PSOM_SIZE="NI NJ NK ngrid"] [PSOM_TAG=name] [PSOM_BUILD_SUFFIX=_x] julia --project=validation validation/difftest.jl <experiment> [nsteps=1] [synced|free] [tol=1e-12]
# Run directory: validation/runs/<experiment>/difftest  (needs the Fortran and Julia namelists; run_case.sh writes them)

module DT
using Libdl, Printf

const FSIZE = Dict(:real8 => 8, :int4 => 4, :int8 => 8, :logical4 => 4)

# ---------------------------------------------------------------- Fortran header parsing
function splitdepth0(s)
    out = String[]; depth = 0; cur = IOBuffer()
    for c in s
        if c == '('; depth += 1
        elseif c == ')'; depth -= 1
        end
        if c == ',' && depth == 0
            push!(out, String(take!(cur)))
        else
            write(cur, c)
        end
    end
    push!(out, String(take!(cur)))
    out
end

"name => fortran type, for every variable declared in the (preprocessed) header module"
function parse_header(bdir)
    txt = read(Cmd(`gfortran -cpp -E -P header.f90`; dir=bdir), String)
    lines = String[]; buf = ""
    for l in split(txt, '\n')
        l = replace(String(l), r"!.*$" => "")
        l = replace(l, r"^\s*&" => "")
        l = rstrip(l)
        if endswith(l, "&")
            buf *= l[1:end-1]; continue
        end
        push!(lines, buf * l); buf = ""
    end
    types = Dict{String,Symbol}()
    for l in lines
        m = nothing
        if occursin("::", l)
            lhs, rest_ = split(l, "::"; limit=2)
            mt = match(r"^\s*(real|integer|logical|character)\b(.*)$"i, lhs)
            mt === nothing || (m = (mt[1], mt[2], rest_))
        end
        if m === nothing
            m = match(r"^\s*(real|integer|logical)\s*(\([^)]*\)|\*\d+)?\s+([A-Za-z_].*)$"i, l)
            m === nothing && continue
        end
        t, attr, rest = lowercase(m[1]), something(m[2], ""), m[3]
        occursin(r"parameter"i, attr) && continue
        ft = t == "real" ? :real8 : t == "integer" ? (occursin("8", attr) ? :int8 : :int4) :
             t == "logical" ? :logical4 : :char
        for item in splitdepth0(rest)
            mm = match(r"^\s*([A-Za-z_][A-Za-z_0-9]*)", item)
            mm === nothing || (types[lowercase(mm[1])] = ft)
        end
    end
    types
end

# ---------------------------------------------------------------- variable mapping
struct Var
    name::String        # fortran name (lowercase)
    jsym::Symbol        # julia global
    ftype::Symbol
    ptr::Ptr{Cvoid}
end

mutable struct Ctx
    lib::Ptr{Cvoid}
    vars::Vector{Var}
    skipped::Vector{String}
end

jdata(x::AbstractArray) = parentmost(x)
parentmost(x::Array) = x
parentmost(x) = parentmost(parent(x))

function build_ctx(libpath, bdir)
    lib = dlopen(libpath)
    types = parse_header(bdir)
    # symbol sizes from address gaps (upper bound)
    syms = Tuple{UInt,String}[]
    for l in eachline(`nm -n $libpath`)
        p = split(l)
        length(p) == 3 || continue
        push!(syms, (parse(UInt, p[1]; base=16), p[3]))
    end
    gap = Dict{String,Int}()
    for i in 1:length(syms)-1
        gap[syms[i][2]] = Int(syms[i+1][1] - syms[i][1])
    end
    prefix = Sys.isapple() ? "___header_MOD_" : "__header_MOD_"
    jl = Dict(lowercase(String(s)) => s for s in names(Main; all=true) if isdefined(Main, s))
    vars = Var[]; skipped = String[]
    for (name, ft) in types
        ft == :char && continue
        haskey(jl, name) || continue
        sym = prefix * name
        haskey(gap, sym) || continue
        x = getfield(Main, jl[name])
        nbytes = x isa AbstractArray ? length(x) * FSIZE[ft] : x isa Number ? FSIZE[ft] : -1
        if nbytes < 0
            push!(skipped, "$name (julia type $(typeof(x)))"); continue
        end
        # gap = distance to next symbol; allow alignment padding only
        if nbytes > gap[sym] || (x isa AbstractArray && gap[sym] - nbytes > 64)
            push!(skipped, "$name (size: julia $nbytes B, fortran ~$(gap[sym]) B)"); continue
        end
        push!(vars, Var(name, jl[name], ft, dlsym(lib, sym[2:end])))
    end
    sort!(vars; by=v -> v.name)
    Ctx(lib, vars, skipped)
end

fptr(c::Ctx, name) = dlsym(c.lib, (Sys.isapple() ? "__header_MOD_" : "__header_MOD_") * lowercase(name))

ctype(ft) = ft == :real8 ? Float64 : ft == :int4 ? Int32 : ft == :int8 ? Int64 : Int32

"read fortran value of v, shaped like the julia data"
function fvalue(v::Var)
    x = getfield(Main, v.jsym)
    T = ctype(v.ftype)
    if x isa AbstractArray
        d = jdata(x)
        return copy(unsafe_wrap(Array, Ptr{T}(v.ptr), size(d)))
    else
        return unsafe_load(Ptr{T}(v.ptr))
    end
end

"copy julia value of v into fortran memory"
function j2f!(v::Var)
    x = getfield(Main, v.jsym)
    T = ctype(v.ftype)
    if x isa AbstractArray
        d = jdata(x)
        f = unsafe_wrap(Array, Ptr{T}(v.ptr), size(d))
        f .= T.(d)
    else
        unsafe_store!(Ptr{T}(v.ptr), T(x))
    end
end

sync!(c::Ctx) = foreach(j2f!, c.vars)

setf!(c::Ctx, name, x::Float64) = unsafe_store!(Ptr{Float64}(fptr(c, name)), x)
setf!(c::Ctx, name, x::Integer) = unsafe_store!(Ptr{Int32}(fptr(c, name)), Int32(x))
function copyf!(c::Ctx, dst, src, n)
    unsafe_copyto!(Ptr{Float64}(fptr(c, dst)), Ptr{Float64}(fptr(c, src)), n)
end

# ---------------------------------------------------------------- fortran calls
fsym(c, name) = dlsym(c.lib, name * "_")
fcall(c, name) = ccall(fsym(c, name), Cvoid, ())
fcall(c, name, a) = ccall(fsym(c, name), Cvoid, (Ptr{Cvoid},), a)
fcall(c, name, a, b) = ccall(fsym(c, name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}), a, b)
fcall(c, name, a, b, d) = ccall(fsym(c, name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), a, b, d)
fcall(c, name, a, b, d, e) = ccall(fsym(c, name), Cvoid, (Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}, Ptr{Cvoid}), a, b, d, e)
I(x) = Ref{Int32}(Int32(x))
R(x) = Ref{Float64}(Float64(x))

# ---------------------------------------------------------------- comparison
struct Diff
    name::String
    rel::Float64
    maxabs::Float64
    scale::Float64
    where::Any
end

function diffs(c::Ctx, extras::Vector{Tuple{String,Any,Any}}=Tuple{String,Any,Any}[]; tol=1e-12)
    out = Diff[]
    function cmp(name, J, F, lo)
        if J isa Number
            d = abs(J - F); s = max(abs(F), abs(J))
            rel = s == 0 ? 0.0 : d / s
            rel > tol && push!(out, Diff(name, rel, d, s, ()))
        else
            d = abs.(J .- F)
            m, idx = findmax(d)
            s = max(maximum(abs, F), maximum(abs, J))
            rel = s == 0 ? 0.0 : m / s
            rel > tol && push!(out, Diff(name, rel, m, s, Tuple(idx) .+ lo .- 1))
        end
    end
    for v in c.vars
        x = getfield(Main, v.jsym)
        J = x isa AbstractArray ? jdata(x) : x
        lo = x isa AbstractArray ? (x isa Array ? ntuple(_ -> 1, ndims(x)) : first.(axes(x))) : ()
        cmp(v.name, J, fvalue(v), lo)
    end
    for (name, J, F) in extras
        cmp(name, J, F, J isa AbstractArray ? ntuple(_ -> 1, ndims(J)) : ())
    end
    sort!(out; by=d -> -d.rel)
end

function report(io, label, ds; maxshow=8)
    if isempty(ds)
        @printf(io, "%-34s ok\n", label)
    else
        @printf(io, "%-34s %d differ:", label, length(ds))
        for d in ds[1:min(end, maxshow)]
            @printf(io, " %s(%.1e @%s)", d.name, d.rel, join(d.where, ","))
        end
        length(ds) > maxshow && print(io, " ...")
        println(io)
    end
end

end # module DT

# ==================================================================== driver
using .DT
using OffsetArrays

const expe = length(ARGS) >= 1 ? ARGS[1] : "wiggle"
const nsteps_test = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1
const mode = length(ARGS) >= 3 ? ARGS[3] : "synced"
const tol = length(ARGS) >= 4 ? parse(Float64, ARGS[4]) : 1e-12

const VAL = @__DIR__
const CODE = joinpath(VAL, "..", "code")
# optional grid-size override, see size_override.sh
const SIZE_SFX = haskey(ENV, "PSOM_SIZE") ? "_" * join(split(ENV["PSOM_SIZE"])[1:3], "x") * "g" * split(ENV["PSOM_SIZE"])[4] : ""
# PSOM_BUILD_SUFFIX selects a variant Fortran build (e.g. _fixwrap, see build_fortran_ref.sh)
const BDIR = joinpath(VAL, "build", expe * SIZE_SFX * get(ENV, "PSOM_BUILD_SUFFIX", ""))
# PSOM_TAG=name uses the namelists of a run_case.sh --tag run (runs/<exp>[_size]_<name>)
const RUN = expe * SIZE_SFX * (haskey(ENV, "PSOM_TAG") ? "_" * ENV["PSOM_TAG"] : "")
const RDIR = joinpath(VAL, "runs", RUN, "difftest")
const EXE = joinpath(CODE, "exe", expe == "model" ? "nh" : "nh_" * expe)

run(`sh $(joinpath(CODE, "tools", "compile_julia.sh")) $expe`)
haskey(ENV, "PSOM_SIZE") && run(`bash -c "source $(joinpath(VAL, "size_override.sh")); write_size_jl $(joinpath(EXE, "inc", "size.jl"))"`)
isfile(joinpath(BDIR, "libpsom_ref.dylib")) || isfile(joinpath(BDIR, "libpsom_ref.so")) || run(`bash $(joinpath(VAL, "build_fortran_ref.sh")) $expe`)
mkpath(RDIR)
cp(joinpath(VAL, "runs", RUN, "fortran", "namelist"), joinpath(RDIR, "namelist"); force=true)
cp(joinpath(VAL, "runs", RUN, "julia", "namelist.yml"), joinpath(RDIR, "namelist.yml"); force=true)
cd(RDIR)

# --- load the Julia model (same as main.jl, without running main)
include(joinpath(EXE, "inc", "includeFile.jl"))
using .fortVar
using .cppdefs

empty!(ARGS); push!(ARGS, joinpath(RDIR, "namelist.yml"))
include(joinpath(EXE, "inc", "ini_param.jl"))

const libpath = joinpath(BDIR, Sys.isapple() ? "libpsom_ref.dylib" : "libpsom_ref.so")
const ctx = DT.build_ctx(libpath, BDIR)
DT.fcall(ctx, "ref_read_params")

const out = open(joinpath(RDIR, "difftest_$(mode).txt"), "w")
function say(args...)
    println(stdout, args...); println(out, args...); flush(out)
end
say("difftest: experiment=$expe mode=$mode nsteps=$nsteps_test tol=$tol")
say("mapped $(length(ctx.vars)) Fortran header variables onto Julia globals")
say("mapped: ", join([v.name for v in ctx.vars], " "))
isempty(ctx.skipped) || say("not mapped: ", join(ctx.skipped, "; "))

const pcorrJ = zeros(rc_kind, maxout)
const pcorrF = zeros(rc_kind, maxout)
const extras = Tuple{String,Any,Any}[]   # per-step returned values: (name, julia Ref, fortran Ref)

# state after reading parameters
DT.report(stdout, "params", DT.diffs(ctx; tol))
DT.report(out, "params", DT.diffs(ctx; tol))

"run one sub-step on both sides and report differences"
function substep(label, jfun, ffun)
    mode == "synced" && (DT.sync!(ctx); copyto!(pcorrF, pcorrJ))
    empty!(extras)
    jfun(); ffun()
    ex = Tuple{String,Any,Any}[(n, j[], f[]) for (n, j, f) in extras]
    ds = DT.diffs(ctx, vcat(ex, [("pcorr", pcorrJ, pcorrF)]); tol)
    DT.report(stdout, label, ds); DT.report(out, label, ds)
    ds
end

ptr(name) = DT.fptr(ctx, name)

"reset the ran3 generator (wiggle linerelax) on both sides to its initial state"
function reset_ran3!()
    isdefined(Main, :ran3_state) && (ran3_state.iff = 0)
    ccall(DT.dlsym(ctx.lib, "ran3_"), Float64, (Ptr{Int32},), Ref{Int32}(-91294))
    isdefined(Main, :ran3) && ran3(Ref(-91294))
end

"compare two arrays and report"
function cmpreport(label, pairs...)
    ds = DT.Diff[]
    for (name, J, F) in pairs
        append!(ds, DT.diffs(DT.Ctx(ctx.lib, DT.Var[], String[]), Tuple{String,Any,Any}[(name, J, F)]; tol))
    end
    DT.report(stdout, label, ds); DT.report(out, label, ds)
end

"unit tests of the routines inside hsolve, on the current (Julia) state"
function unit_hsolve(dtim)
    chJ = zeros(9, NI, NJ); rhsJ = zeros(NI, NJ); chF = zeros(9, NI, NJ); rhsF = zeros(NI, NJ)
    chfine(dtim, chJ, rhsJ, HOST); DT.fcall(ctx, "chfine", DT.R(dtim), chF, rhsF)
    cmpreport("  unit chfine", ("ch", chJ, chF), ("rhs", rhsJ, rhsF))
    hJ = copy(h); hF = copy(parent(h))
    hsums = zeros(NI, 12); hsolve_sums!(hsums, HOST)
    hfill(dtim, hJ, hsums, HOST); DT.fcall(ctx, "hfill", DT.R(dtim), hF)
    cmpreport("  unit hfill", ("h", parent(hJ), hF))
    hJ = copy(h); hF = copy(parent(h))
    mprove(hJ, chJ, rhsJ, dtim, HOST); DT.fcall(ctx, "mprove", hF, chJ, rhsJ, DT.R(dtim))
    cmpreport("  unit mprove", ("h", parent(hJ), hF))
end
"unit test of conadjust on a state made statically unstable in every other column, with a tracer that
varies in i, j and k (the short test runs never convect); the Julia state is restored afterwards"
function unit_conadjust(step)
    saved = [(x, copy(x)) for x in (s, T, rho, Tr, conv, con100)]
    for j in 1:NJ, i in 1:NI, k in 1:NK
        isodd(i + j) && (rho[i, j, k] -= 1e-2 * (NK - k))
        for it in 1:ntr
            Tr[it, i, j, k, 0] = k + 0.1 * i + 0.01 * j + it
        end
    end
    ds = substep("  unit conadjust (unstable)", () -> conadjust(step, 0), () -> DT.fcall(ctx, "conadjust", DT.I(step), DT.I(0)))
    say("    columns convecting: ", count(==(1), conv), " cells with conv = 1")
    for (x, c) in saved
        copyto!(x, c)
    end
    ds
end

"Fortran header array as a Julia array with the same axes as the mapped Julia global (aliases Fortran memory)"
function fwrap(name)
    v = ctx.vars[findfirst(v -> v.name == name, ctx.vars)]
    x = getfield(Main, v.jsym)
    a = unsafe_wrap(Array, Ptr{Float64}(v.ptr), size(DT.jdata(x)))
    x isa OffsetArray ? OffsetArray(a, axes(x)...) : a
end

pview(p) = OffsetArrays.reshape(view(p, 1:(NI+2)*(NJ+2)*(NK+2)), 0:NI+1, 0:NJ+1, 0:NK+1)

# ------------------------------------------------------------------ initialisation (main.jl / main.f90)
say("\n=== initialisation")
substep("ini_setup", () -> ini_setup(pcorrJ), () -> DT.fcall(ctx, "ini_setup", pcorrF))
substep("meanh", () -> (global hmean = meanh(NI, NJ, h)),
        () -> DT.fcall(ctx, "meanh", DT.I(NI), DT.I(NJ), ptr("h"), ptr("hmean")))
substep("sigma", () -> sigma(), () -> DT.fcall(ctx, "sigma"))
substep("staticsigma", () -> staticsigma(), () -> DT.fcall(ctx, "staticsigma"))
substep("tracerinit", () -> tracerinit(0), () -> DT.fcall(ctx, "tracerinit", DT.I(0)))
substep("hsave", () -> hsave(), () -> DT.fcall(ctx, "hsave"))
substep("ini_uv", () -> ini_uv(0), () -> DT.fcall(ctx, "ini_uv", DT.I(0)))
let fJ = facediv(EPS), fF = DT.R(0)
    DT.fcall(ctx, "facediv", DT.R(EPS), fF); say("  facediv(EPS): J=$fJ F=$(fF[])")
end
substep("vort", () -> vort(0), () -> DT.fcall(ctx, "vort", DT.I(0)))
substep("diag_n2", () -> diag_n2(), () -> DT.fcall(ctx, "diag_n2"))
substep("diag_n2budget", () -> diag_n2budget(0), () -> DT.fcall(ctx, "diag_n2budget", DT.I(0)))
substep("setbc", () -> setbc(0), () -> DT.fcall(ctx, "setbc", DT.I(0)))
substep("correctbc", () -> correctbc(), () -> DT.fcall(ctx, "correctbc"))
substep("checks", () -> checks(), () -> DT.fcall(ctx, "checks"))

# ------------------------------------------------------------------ time steps (momentum split up)
for step in 1:nsteps_test
    say("\n=== step $step")
    global time_nondim = dtf * step
    global time_seconds = time_nondim * TL
    DT.setf!(ctx, "time_nondim", time_nondim); DT.setf!(ctx, "time_seconds", time_seconds)

    substep("diag_n2", () -> diag_n2(), () -> DT.fcall(ctx, "diag_n2"))
    substep("momentum: oldh=h, rho_old=rho", () -> (oldh .= h; rho_old .= rho),
            () -> (DT.copyf!(ctx, "oldh", "h", (NI+2)*(NJ+2)); DT.copyf!(ctx, "rho_old", "rho", (NI+2)*(NJ+2)*(NK+2))))
    for ivb in 1:3
        ivs = ivb == 1 ? 0 : 1
        ivf = ivb == 3 ? 0 : 1
        dtim = dtf / (ivb == 1 ? 3e0 : (ivb == 2 ? 2 : 1))
        edt = EPS / dtim
        setfiv() = (DT.setf!(ctx, "ivb", ivb); DT.setf!(ctx, "ivs", ivs); DT.setf!(ctx, "ivf", ivf); DT.setf!(ctx, "tsp", dtim * 1e5))
        F(name, args...) = () -> (setfiv(); DT.fcall(ctx, name, args...))
        t = "s$step.$ivb "
        substep(t * "findzall", () -> findzall(), F("findzall"))
        substep(t * "sigma", () -> sigma(), F("sigma"))
        substep(t * "advection_and_mixing", () -> advection_and_mixing(ivs, ivf, dtim, step),
                F("advection_and_mixing", DT.I(ivs), DT.I(ivf), DT.R(dtim), DT.I(step)))
        substep(t * "intpol", () -> intpol(), F("intpol"))
        substep(t * "rpevalgrad_Song", () -> rpevalgrad_Song(ivf), F("rpevalgrad_song", DT.I(ivf)))
        if use_shchepetkin != 0
            substep(t * "rpevalgrad_Sche", () -> rpevalgrad_Sche(ivf), F("rpevalgrad_sche", DT.I(ivf)))
            substep(t * "Sche copies",
                () -> (@views drpx[1:NI, 1:NJ, 1:NK] .= ru4_Sche[1:NI, 1:NJ, 1:NK];
                       @views drpy[1:NI, 1:NJ, 1:NK] .= rv4_Sche[1:NI, 1:NJ, 1:NK];
                       @views grpifc[0:NI, 1:NJ, 1:NK] .= ru2_Sche[0:NI, 1:NJ, 1:NK];
                       @views grpjfc[1:NI, 0:NJ, 1:NK] .= rv2_Sche[1:NI, 0:NJ, 1:NK]),
                () -> (@views fwrap("drpx")[1:NI, 1:NJ, 1:NK] .= fwrap("ru4_sche")[1:NI, 1:NJ, 1:NK];
                       @views fwrap("drpy")[1:NI, 1:NJ, 1:NK] .= fwrap("rv4_sche")[1:NI, 1:NJ, 1:NK];
                       @views fwrap("grpifc")[0:NI, 1:NJ, 1:NK] .= fwrap("ru2_sche")[0:NI, 1:NJ, 1:NK];
                       @views fwrap("grpjfc")[1:NI, 0:NJ, 1:NK] .= fwrap("rv2_sche")[1:NI, 0:NJ, 1:NK]))
        end
        substep(t * "coriolis", () -> coriolis(ivs), F("coriolis", DT.I(ivs)))
        substep(t * "srcface", () -> srcface(ivs, step), F("srcface", DT.I(ivs), DT.I(step)))
        mode == "synced" && (DT.sync!(ctx); unit_hsolve(dtim))
        substep(t * "hsolve", () -> hsolve(h, oldh, hdt, dtim), F("hsolve", ptr("h"), ptr("oldh"), ptr("hdt"), DT.R(dtim)))
        substep(t * "calcskfc", () -> calcfkfc(), F("calcskfc"))
        substep(t * "vhydro", () -> vhydro(dtim), F("vhydro", DT.R(dtim)))
        cJ = Ref(0e0); cF = DT.R(0)
        substep(t * "cfdiv", () -> (cfdiv(cJ); push!(extras, ("cfcdiv", cJ, cF))), F("cfdiv", cF))
        cfF = mode == "synced" ? cJ[] : cF[]
        substep(t * "newcor", () -> newcor(dtim, ivs), F("newcor", DT.R(dtim), DT.I(ivs)))
        substep(t * "newsrc", () -> newsrc(), F("newsrc"))
        substep(t * "mgrid", () -> (mode == "synced" && reset_ran3!(); mgrid(pcorrJ, dtim, edt, cJ[])), F("mgrid", pcorrF, DT.R(dtim), DT.R(edt), DT.R(cfF)))
        substep(t * "vface", () -> vface(pcorrJ, dtim), F("vface", pcorrF, DT.R(dtim)))
        substep(t * "vcenter", () -> vcenter(pview(pcorrJ), dtim, ivf), F("vcenter", pcorrF, DT.R(dtim), DT.I(ivf)))
        if fnhhy != 0e0
            substep(t * "pcorrect", () -> pcorrect(pview(pcorrJ)), F("pcorrect", pcorrF))
        end
        fJ = Ref(0e0); fF = DT.R(0)
        substep(t * "facediv", () -> (fJ[] = facediv(dtim); push!(extras, ("fdiv", fJ, fF))), F("facediv", DT.R(dtim), fF))
        dJ = Ref(0e0); dF = DT.R(0)
        substep(t * "cdiv", () -> (dJ[] = cdiv(dtim, ivf); push!(extras, ("ctrdiv", dJ, dF))), F("cdiv", DT.R(dtim), dF, DT.I(ivf)))
    end
    substep("s$step evalrho", () -> evalrho(rho, 0), () -> DT.fcall(ctx, "evalrho", ptr("rho"), DT.I(0)))
    # experiment-specific end of momentum (see code/<exp>/src/momentum.f90)
    if expe == "waves"   # waves: no conadjust; sponge after diag_n2 with the last RK dtim (= dtf)
        substep("s$step diag_n2", () -> diag_n2(), () -> DT.fcall(ctx, "diag_n2"))
        substep("s$step sponge", () -> sponge(0, dtf), () -> DT.fcall(ctx, "sponge", DT.I(0), DT.R(dtf)))
    else
        mode == "synced" && unit_conadjust(step)
        substep("s$step conadjust", () -> conadjust(step, 0), () -> DT.fcall(ctx, "conadjust", DT.I(step), DT.I(0)))
        substep("s$step diag_n2", () -> diag_n2(), () -> DT.fcall(ctx, "diag_n2"))
    end
    substep("s$step diag_n2budget", () -> diag_n2budget(step), () -> DT.fcall(ctx, "diag_n2budget", DT.I(step)))
    substep("s$step meanh", () -> (global hmean = meanh(NI, NJ, h)),
            () -> DT.fcall(ctx, "meanh", DT.I(NI), DT.I(NJ), ptr("h"), ptr("hmean")))
end
close(out)
