###
# Here we implement inversion of banded matrices, beginning with Bidiagonal
###

function SemiseparableMatrix(Bi::ApplyMatrix{T,typeof(inv),<:Tuple{Bidiagonal}}) where T
    B = inv(Bi)
    n = size(B,1)
    ℓ = [1; cumprod(-B.ev./B.dv[1:end-1])]
    if B.uplo == 'L'
        SemiseparableMatrix(ApplyArray(*, inv.(B.dv) .* ℓ, inv.(ℓ)'), Zeros{T}(n,n), 0, 1)
    else
        SemiseparableMatrix(Zeros{T}(n,n), ApplyArray(*, inv.(ℓ), (inv.(B.dv) .* ℓ)'), 1, 0)
    end
end

###
# Givens-weight representation of inv(B), B upper triangular with bandwidth m:
#
#   X[i,j] = w_iᵀ S_i ⋯ S_{j-1} c_j  (i ≤ j),   [c_i S_i] = (R_{i,1} ⋯ R_{i,m})[1:m, :],
#
# where R_{i,k} is a rotation acting on coordinates (k, m+1) with cosine c[k,i]
# and sine s[k,i]. In particular c_i = c[1,i] e_1. For m = 1 this is
# X[i,j] = t_i c_j s_i ⋯ s_{j-1}.
###

struct GivensWeightMatrix{T} <: AbstractMatrix{T}
    c::Matrix{T}  # m × n cosines
    s::Matrix{T}  # m × n sines
    w::Matrix{T}  # m × n weights, w[:,i] = w_i
end

size(X::GivensWeightMatrix) = (size(X.w,2), size(X.w,2))

function getindex(X::GivensWeightMatrix{T}, i::Int, j::Int) where T
    i > j && return zero(T)
    m = size(X.w,1)
    u = [X.w[:,i]; zero(T)]
    for k = i:j-1 # [uᵀ 0] G_k = [uᵀc_k  uᵀS_k], keep uᵀS_k
        for p = 1:m
            u[p], u[m+1] = X.c[p,k]*u[p] - X.s[p,k]*u[m+1], X.s[p,k]*u[p] + X.c[p,k]*u[m+1]
        end
        u = [u[2:end]; zero(T)]
    end
    X.c[1,j] * u[1]
end

# Fill row-by-row, propagating u = w_iᵀ S_i ⋯ S_{j-1} in place across j: O(n² m) total
# with O(m) workspace, instead of O(n m) per entry via getindex.
function copyto!(dest::StridedMatrix, X::GivensWeightMatrix{T}) where T
    size(dest) == size(X) || throw(DimensionMismatch("size of destination $(size(dest)) does not match $(size(X))"))
    n = size(X,1)
    m = size(X.w,1)
    c, s, w = X.c, X.s, X.w
    u = zeros(T, m+1)
    for i = 1:n
        for p = 1:m
            u[p] = w[p,i]
        end
        u[m+1] = zero(T)
        for j = 1:i-1
            dest[i,j] = zero(T)
        end
        for j = i:n
            dest[i,j] = c[1,j] * u[1]
            j == n && break
            for p = 1:m # [uᵀ 0] G_j = [uᵀc_j  uᵀS_j]
                up, um = u[p], u[m+1]
                u[p], u[m+1] = c[p,j]*up - s[p,j]*um, s[p,j]*up + c[p,j]*um
            end
            for p = 1:m # shift: keep uᵀS_j
                u[p] = u[p+1]
            end
            u[m+1] = zero(T)
        end
    end
    dest
end

Base.Matrix{T}(X::GivensWeightMatrix) where T = copyto!(Matrix{T}(undef, size(X)...), X)
Base.Matrix(X::GivensWeightMatrix{T}) where T = Matrix{T}(X)
Base.Array(X::GivensWeightMatrix) = Matrix(X)

# Broadcasting: densify any GivensWeightMatrix arguments via the fast copyto! (rather than
# entrywise getindex) and re-broadcast. This also covers A ± X for any AbstractMatrix A,
# since +/- on arrays lower to broadcast.
struct GivensWeightStyle <: Broadcast.AbstractArrayStyle{2} end
GivensWeightStyle(::Val{2}) = GivensWeightStyle()
GivensWeightStyle(::Val{N}) where N = Broadcast.DefaultArrayStyle{N}()
Broadcast.BroadcastStyle(::Type{<:GivensWeightMatrix}) = GivensWeightStyle()
Broadcast.BroadcastStyle(::GivensWeightStyle, ::Broadcast.AbstractArrayStyle{2}) = GivensWeightStyle()
Broadcast.BroadcastStyle(::GivensWeightStyle, ::Broadcast.DefaultArrayStyle{2}) = GivensWeightStyle()

_givensweight_densify(X::GivensWeightMatrix) = Matrix(X)
_givensweight_densify(x) = x
Broadcast.broadcasted(::GivensWeightStyle, f, args...) = Broadcast.broadcasted(f, map(_givensweight_densify, args)...)

# O(n m²) construction, bottom-up. T_{i+1} is the upper triangular m × m factor of the
# Gram matrix of rows i+1:i+m of inv(B ⊕ I_m) (so T_{n+1} = I), and
# M_i = [e_1/d_i | L_i T_{i+1}] with L_i the companion matrix of row i of B.
# Rotating columns (k, m+1) of M_i for k = m:-1:1 gives [T_i 0]; w_i = T_i[1,:].
function GivensWeightMatrix(Bi::ApplyMatrix{<:Any,typeof(inv),<:Tuple{BandedMatrix}})
    B = only(Bi.args)
    bandwidth(B,1) ≤ 0 || throw(ArgumentError("matrix must be upper triangular"))
    n, m = size(B,1), max(bandwidth(B,2), 1)
    T = float(eltype(B))
    c, s, w = zeros(T,m,n), zeros(T,m,n), zeros(T,m,n)
    R = Matrix{T}(I, m, m)
    for i = n:-1:1
        d = B[i,i]
        b = [i+k ≤ n ? B[i,i+k] : zero(T) for k = 1:m]
        M = [inv(d) -b'R/d; zeros(T,m-1) R[1:m-1,:]]
        for k = m:-1:1
            r = hypot(M[k,k], M[k,m+1])
            c[k,i], s[k,i] = M[k,k]/r, M[k,m+1]/r
            M[:,k], M[:,m+1] = c[k,i]*M[:,k] + s[k,i]*M[:,m+1], c[k,i]*M[:,m+1] - s[k,i]*M[:,k]
        end
        R = M[:,1:m]
        w[:,i] = R[1,:]
    end
    GivensWeightMatrix(c, s, w)
end