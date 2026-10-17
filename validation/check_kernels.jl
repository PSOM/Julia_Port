# Static check of the KernelAbstractions kernels (see GPU_PORTING.md, rule 2): a kernel body must not use
# header arrays or non-const globals by name (fine on the CPU, but they don't exist on a GPU).
#
# Usage: julia validation/check_kernels.jl [experiment ...]     (default: model wiggle waves)
# Scans code/model/src/*.jl and code/<experiment>/src/*.jl. Exit status 1 if anything is found.

const CODE = joinpath(@__DIR__, "..", "code")
exps = isempty(ARGS) ? ["model", "wiggle", "waves"] : ARGS

# Names that must not appear in kernels: every name defined in header.jl except const scalars with a
# literal value, plus everything assigned in ini_param.jl (non-const globals).
function forbidden_names(hdr_files, param_file)
    bad = Set{Symbol}()
    for f in hdr_files
        isfile(f) || continue
        for ex in Meta.parseall(read(f, String)).args
            ex isa Expr || continue
            e = ex
            isconst = false
            while e isa Expr && e.head in (:global, :const)
                e.head == :const && (isconst = true)
                e = e.args[1]
            end
            (e isa Expr && e.head == :(=)) || continue
            lhs, rhs = e.args
            name = lhs isa Symbol ? lhs : (lhs isa Expr && lhs.head == :(::) ? lhs.args[1] : nothing)
            name isa Symbol || continue
            literal = rhs isa Number || rhs isa Bool || (rhs isa Expr && rhs.head == :call && rhs.args[1] in (:*, :/, :+, :-) &&
                                                         all(a -> a isa Number || a isa Symbol, rhs.args[2:end]))
            (isconst && literal) || push!(bad, name)
        end
    end
    for ex in Meta.parseall(read(param_file, String)).args
        ex isa Expr && ex.head == :(=) && ex.args[1] isa Symbol && push!(bad, ex.args[1])
    end
    delete!(bad, :rc_kind)
    return bad
end

# names of the header arrays (const globals assigned something that is not a literal)
function header_arrays(hdr_files)
    out = Set{Symbol}()
    for f in hdr_files
        isfile(f) || continue
        for m in eachmatch(r"^\s*global\s+const\s+([A-Za-z_][A-Za-z_0-9]*)\s*=\s*(OffsetArray|@fortGetArr|zeros)"m, read(f, String))
            push!(out, Symbol(m.captures[1]))
        end
    end
    return out
end

argname(a) = a isa Symbol ? a :
             a isa Expr && a.head == :(::) ? argname(a.args[1]) :
             a isa Expr && a.head == :macrocall ? argname(a.args[end]) :
             a isa Expr && a.head == :kw ? argname(a.args[1]) : nothing

# symbols assigned anywhere in the body (locals, loop variables, destructuring)
function assigned!(set, ex)
    ex isa Expr || return set
    if ex.head in (:(=), :local) || (ex.head == :for)
        tgt = ex.head == :local ? ex.args : [ex.args[1]]
        for t in tgt
            lhs = t isa Expr && t.head == :(=) ? t.args[1] : t
            lhs isa Expr && lhs.head == :block && (foreach(x -> assigned!(set, Expr(:(=), x.args[1], 0)), lhs.args); continue)
            for s in (lhs isa Symbol ? [lhs] : lhs isa Expr && lhs.head in (:tuple, :(::)) ? lhs.args : [])
                s isa Symbol && push!(set, s)
                s isa Expr && s.head == :(::) && s.args[1] isa Symbol && push!(set, s.args[1])
            end
        end
    end
    foreach(a -> assigned!(set, a), ex.args)
    return set
end

# symbols used in the body, skipping field names (a.b) and macro names
function used!(set, ex)
    if ex isa Symbol
        push!(set, ex)
    elseif ex isa Expr
        if ex.head == :.
            used!(set, ex.args[1])
        elseif ex.head == :macrocall
            foreach(a -> used!(set, a), ex.args[2:end])
        else
            foreach(a -> used!(set, a), ex.args)
        end
    end
    return set
end

function kernels(ex, out=[])
    if ex isa Expr
        if ex.head == :macrocall && ex.args[1] == Symbol("@kernel")
            push!(out, ex.args[end])
        else
            foreach(a -> kernels(a, out), ex.args)
        end
    end
    return out
end

# kernel launches: launch!(S, k, ranges, args...) and k(backend, ...)(args...; ndrange): every argument
# that is a header array must be passed as S.name (or another array set), never by its global name
function launch_args(ex, out=[])
    if ex isa Expr
        if ex.head == :call && ex.args[1] == :launch!
            append!(out, [(a, "launch!") for a in ex.args[5:end]])
        elseif ex.head == :call && ex.args[1] isa Expr && ex.args[1].head == :call &&
               ex.args[1].args[1] isa Symbol && startswith(string(ex.args[1].args[1]), "k_")
            append!(out, [(a, string(ex.args[1].args[1])) for a in ex.args[2:end] if !(a isa Expr && a.head == :parameters)])
        elseif ex.head == :macrocall && ex.args[1] == Symbol("@kernel")
            return out
        end
        foreach(a -> launch_args(a, out), ex.args)
    end
    return out
end
# function definitions (long and short form), outside kernels
function functions(ex, out=[])
    if ex isa Expr
        if ex.head == :macrocall && ex.args[1] == Symbol("@kernel")
            return out
        elseif ex.head == :function || (ex.head == :(=) && ex.args[1] isa Expr && ex.args[1].head == :call)
            push!(out, ex)
            return out
        end
        foreach(a -> functions(a, out), ex.args)
    end
    return out
end

nbad = 0
for exp in exps
    hdr = [joinpath(CODE, "model", "src", "header.jl"), joinpath(CODE, exp, "src", "header.jl")]
    bad = forbidden_names(hdr, joinpath(CODE, "model", "inc", "ini_param.jl"))
    hdrarrays = header_arrays(hdr)
    dirs = exp == "model" ? [joinpath(CODE, "model", "src")] : [joinpath(CODE, exp, "src")]
    for d in dirs, f in sort(filter(endswith(".jl"), readdir(d)))
        for k in kernels(Meta.parseall(read(joinpath(d, f), String)))
            sig, body = k.args[1], k.args[2]
            name = sig.args[1]
            args = Set(filter(!isnothing, map(argname, sig.args[2:end])))
            locals = assigned!(Set{Symbol}(), body)
            hits = sort(collect(intersect(setdiff(used!(Set{Symbol}(), body), args, locals), bad)))
            if !isempty(hits)
                global nbad += 1
                println("$(joinpath(basename(d) == "src" ? basename(dirname(d)) : d, f)): kernel $name uses globals: ", join(hits, ", "))
            end
        end
        for fdef in functions(Meta.parseall(read(joinpath(d, f), String)))
            sig, body = fdef.args[1], fdef.args[2]
            sig isa Expr && sig.head == :where && (sig = sig.args[1])
            fargs = sig isa Expr && sig.head == :call ? Set(filter(!isnothing, map(argname, sig.args[2:end]))) : Set{Symbol}()
            flocals = assigned!(Set{Symbol}(), body)
            for (a, k) in launch_args(body)
                if a isa Symbol && a in hdrarrays && !(a in fargs) && !(a in flocals)
                    global nbad += 1
                    println("$(joinpath(basename(dirname(d)), f)): $k is passed the host array `$a` (use S.$a)")
                end
            end
        end
    end
end
println(nbad == 0 ? "check_kernels: OK" : "check_kernels: $nbad kernel(s) use globals")
exit(nbad == 0 ? 0 : 1)
