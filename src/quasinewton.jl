export BFGS
"""
    BFGS

Quasi-Newton descent method.
"""
mutable struct BFGS{T<:AbstractFloat,
                    V<:AbstractVector{T},
                    C<:Cholesky{T}} <: CoreMethod
    hess::C
    x::V
    g::V
    xpre::V
    gpre::V
    d::V
    xdiff::V
    gdiff::V
    y::T
end

@inline gradientvec(M::BFGS) = M.g
@inline argumentvec(M::BFGS) = M.x
@inline step_origin(M::BFGS) = M.xpre


function BFGS(x::AbstractVector{T}) where {T}
    F = float(T)
    n = length(x)
    m = similar(x, F, (n, n))
    for j in 1:n, i in 1:n
        m[i,j] = (i == j)
    end
    cm = cholesky!(m)
    bfgs = BFGS(cm,
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                similar(x, F),
                zero(T)
               )
    return bfgs
end

function init!(M::BFGS{T}, optfn!, x0) where {T}
    optfn!(x0, zero(T), x0)
    copy!(M.xpre, M.x)
    copy!(M.gpre, M.g)
    map!(-, M.d, M.g)
    α = strong_backtracking!(optfn!, M.xpre, M.d, M.y, M.gpre, α = 1e-4, β = 0.01, σ = 0.9)
    M.xdiff .= M.x - M.xpre
    M.gdiff .= M.g - M.gpre

    scale = dot(M.gdiff, M.gdiff) / dot(M.xdiff, M.gdiff)
    H = M.hess.factors
    nr, nc = size(H)
    for j in 1:nc, i in 1:nr
        H[i, j] = (i == j) * sqrt(abs(M.gdiff[i] / M.xdiff[j]))
    end
    return
end

@inline function reset!(M::BFGS)
    H = M.hess.factors
    nr, nc = size(H)
    for j in 1:nc, i in 1:nr
        H[i, j] = (i == j)
    end
    return
end

function reset!(M::BFGS, x0, scale::Real=1)
    copy!(M.x, x0)
    H = M.hess.factors
    nr, nc = size(H)
    for j in 1:nc, i in 1:nr
        H[i, j] = (i == j) * sqrt(scale)
    end
    return
end

@inline function callfn!(M::BFGS, fdf, x, α, d)
    __update_arg!(M, x, α, d)
    y, g = fdf(M.x, M.g)
    __update_grad!(M, g)
    M.y = y
    return y, g
end

function __descent_dir!(M::BFGS)
    ldiv!(M.d, M.hess, M.gpre)
    lmul!(-1, M.d)
    return M.d
end

@inline function __step_init!(M::BFGS, optfn!)
    #=
    argument and gradient from the end of the last
    iteration are stored into `xpre` and `gpre`
    =#
    M.gpre, M.g = M.g, M.gpre
    M.xpre, M.x = M.x, M.xpre
    return
end

function __compute_step!(M::BFGS, optfn!, d, maxstep)
    x, xpre, g, gpre, invH = M.x, M.xpre, M.g, M.gpre, M.invH
    α = strong_backtracking!(optfn!, xpre, d, M.y, gpre, αmax = maxstep, β = 0.01, σ = 0.9)
    #=
    BFGS update:
             Hδδ'H    γγ'
    H <- H - ------ + ---
              δ'Hδ    δ'γ
    =#
    δ, γ = M.xdiff, M.gdiff
<<<<<<< Updated upstream
    map!(-, γ, g, gpre)
    map!(-, δ, x, xpre)
    denom = dot(δ, γ)
    δscale = 1 + dot(γ, invH, γ) / denom
    # d <- B * γ
    mul!(d, invH, γ, 1, 0)
    invH .= invH .- (δ .* d' .+ d .* δ') ./ denom .+ δscale .* δ .* δ' ./ denom
=======
    γ .= M.g .- M.gpre
    δ .= M.x .- M.xpre

    #=
    H = H.U' * H.U
    d <- H.U * δ
    δ'Hδ = d ⋅ d
    =#
    mul!(M.d, M.hess.U, δ)
    d1 = dot(M.d, M.d)
    d2 = dot(δ, γ)
    # δ <- H.U' * H.U * δ = H * δ
    mul!(δ, M.hess.U', M.d)
    rdiv!(δ, sqrt(d1))
    rdiv!(γ, sqrt(d2))
    lowrankupdate!(M.hess, γ)
    lowrankdowndate!(M.hess, δ)
>>>>>>> Stashed changes
    return α
end

@inline function __update_arg!(M::BFGS, x, α, d)
    map!(M.x, d, x) do a, b
        muladd(α, a, b)
    end
    return
end

@inline function __update_arg!(M::BFGS, x)
    if x !== M.x
        copy!(M.x, x)
    end
    return
end

@inline function __update_grad!(M::BFGS, g)
    if M.g !== g
        copy!(M.g, g)
    end
    return
end