# Compare Julia and Fortran PSOM full-state dumps (op.pickup.<step>.bin, Fortran w_pickup layout).
#
# Usage: julia compare.jl <fortran_dir> <julia_dir> NI NJ NK [ntr=1] [steps=all common]
#
# For each step and field prints:
#   maxabs  = max |J - F|
#   rel     = maxabs / max |F|            (normalised by the field magnitude)
#   where   = (i,j,k,...) of maxabs in the model's own index convention (0-based halos where Fortran has them)
#   int     = same relative error restricted to interior points (halo rows i=0,NI+1 / j=0,NJ+1 / k=0,NK+1 removed)

using Printf

function layout(NI, NJ, NK, ntr)
    # name, dims, lower index bounds (as declared in header.f90)
    [("h",      (NI+2, NJ+2),             (0, 0)),
     ("u",      (NI+2, NJ+2, NK+2, 2),    (0, 0, 0, 0)),
     ("v",      (NI+2, NJ+2, NK+2, 2),    (0, 0, 0, 0)),
     ("w",      (NI+2, NJ+2, NK+2, 2),    (0, 0, 0, 0)),
     ("uf",     (NI+1, NJ, NK),           (0, 1, 1)),
     ("vf",     (NI, NJ+1, NK),           (1, 0, 1)),
     ("wf",     (NI, NJ, NK+1),           (1, 1, 0)),
     ("T",      (NI+2, NJ+2, NK+2, 2),    (0, 0, 0, 0)),
     ("S",      (NI+2, NJ+2, NK+2, 2),    (0, 0, 0, 0)),
     ("Tr",     (ntr, NI+2, NJ+2, NK+2, 2), (1, 0, 0, 0, 0)),
     ("p",      (NI+2, NJ+2, NK+2),       (0, 0, 0)),
     ("gradhn", (NI+2, NJ+2, 2),          (0, 0, 1)),
     ("hxn",    (NI+1, NJ, NK),           (0, 1, 1)),
     ("hyn",    (NI, NJ+1, NK),           (1, 0, 1))]
end

# interior index ranges: strip one halo cell on each side of every 0:N+1 dimension
function interior(dims, lo, NI, NJ, NK)
    ntuple(length(dims)) do d
        (lo[d] == 0 && dims[d] in (NI+2, NJ+2, NK+2)) ? (2:dims[d]-1) : (1:dims[d])
    end
end

function readdump(file, lay)
    raw = reinterpret(Float64, read(file))
    out = Dict{String,Array{Float64}}()
    pos = 0
    for (name, dims, _) in lay
        n = prod(dims)
        out[name] = reshape(raw[pos+1:pos+n], dims)
        pos += n
    end
    pos == length(raw) || error("$file: expected $pos doubles, found $(length(raw))")
    out
end

stepof(f) = parse(Int, match(r"op\.pickup\.(\d+)\.bin$", f)[1])
dumps(dir) = Dict(stepof(f) => joinpath(dir, f) for f in readdir(dir) if occursin(r"^op\.pickup\.\d+\.bin$", f))

function compare(fdir, jdir, NI, NJ, NK, ntr=1; steps=nothing, io=stdout)
    lay = layout(NI, NJ, NK, ntr)
    fd, jd = dumps(fdir), dumps(jdir)
    common = sort(collect(intersect(keys(fd), keys(jd))))
    steps === nothing || (common = filter(in(steps), common))
    isempty(common) && error("no common dump steps")
    worst = Dict{String,Float64}()
    for st in common
        F, J = readdump(fd[st], lay), readdump(jd[st], lay)
        @printf(io, "step %d\n", st)
        for (name, dims, lo) in lay
            d = abs.(J[name] .- F[name])
            I = interior(dims, lo, NI, NJ, NK)
            mi = maximum(d[I...])
            m, idx = findmax(d)
            scale = maximum(abs, F[name])
            rel = scale == 0 ? m : m / scale
            reli = scale == 0 ? mi : mi / scale
            worst[name] = max(get(worst, name, 0.0), reli)
            where = Tuple(idx) .+ lo .- 1
            flag = reli > 1e-10 ? "  <-- interior" : (rel > 1e-10 ? "  <-- halo only" : "")
            @printf(io, "  %-7s rel=%10.3e  int=%10.3e  maxabs=%10.3e  max|F|=%10.3e  at %s%s\n", name, rel, reli, m, scale, where, flag)
        end
    end
    @printf(io, "\nworst interior rel error over all steps:\n")
    for (name, _, _) in lay
        @printf(io, "  %-7s %10.3e\n", name, worst[name])
    end
    worst
end

if abspath(PROGRAM_FILE) == @__FILE__
    fdir, jdir = ARGS[1], ARGS[2]
    NI, NJ, NK = parse.(Int, ARGS[3:5])
    ntr = length(ARGS) >= 6 ? parse(Int, ARGS[6]) : 1
    steps = length(ARGS) >= 7 ? eval(Meta.parse(ARGS[7])) : nothing
    compare(fdir, jdir, NI, NJ, NK, ntr; steps)
end
