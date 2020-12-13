export optimize!

"""
    optimize!(M::DescentWrapper, fdf, x0)

Find an optimizer for `fdf`, starting with the initial approximation `x0`. 
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite 
it with the gradient.
"""
function optimize!(M::Wrapper, fdf, x0)
    optfn!(x, α, d) = callfn!(M, fdf, x, α, d)

    init!(M, optfn!, x0)
    while !stopcond(M)
        step!(M, optfn!)
    end
    return convstat(M)
end

"""
    optimize!(M::CoreMethod, fdf, x0; gtol = 1e-6, maxiter = 100)

Find an optimizer for `fdf`, starting with the initial approximation `x0`. 
`fdf(x, g)` must return a tuple (f(x), ∇f(x)) and, if `g` is mutable, overwrite 
it with the gradient.
"""
function optimize!(M::CoreMethod, fdf, x0; gtol = convert(eltype(x0), 1e-6), maxiter = 100, maxcalls = nothing)
    if !isnothing(gtol) && gtol > 0
        M = StopByGradient(M, gtol)
    end
    if isnothing(maxiter) || maxiter < 0
        M = LimitIters(M)
    else
        M = LimitIters(M, maxiter)
    end
    if isnothing(maxcalls) || maxcalls < 0
        M = LimitCalls(M)
    else
        M = LimitCalls(M, maxcalls)
    end
    optimize!(M, fdf, x0)
end