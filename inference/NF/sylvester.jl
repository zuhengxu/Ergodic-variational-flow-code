using Bijectors, LinearAlgebra, LogExpFunctions
using Flux:Functors
import Bijectors
################################################################################
#                            Sylvester Flows                                   #
#             Ref: https://arxiv.org/pdf/1803.05649.pdf                        #
################################################################################

###############
# Sylvester #
###############

struct SylvesterLayer{T1<:AbstractMatrix{<:Real}, T2<:AbstractMatrix{<:Real}, T3<:Union{Real, AbstractVector{<:Real}}} <: Bijector{1}
    r1::T1 # upper triangular
    r2::T1 # upper triangular
    h::T2
    b::T3
end
function Base.:(==)(b1::SylvesterLayer, b2::SylvesterLayer)
    return b1.r1 == b2.r1 && b1.r2 == b2.r2 && b1.q == b2.q && bq.b == b2.b
end

function SylvesterLayer(dims::Int, H::Int, wrapper=identity)
    @assert H > 0 && H ≤ dims 
    # need to ensure diag(r1.*r2)  > -1 and r1, r2 upper diag 
    r1 = wrapper(UpperTriangular(randn(dims, dims)))
    r2 = wrapper(UpperTriangular(randn(dims, dims)))
    h = wrapper(randn(dims, H))
    b = wrapper(randn(dims))
    return SylvesterLayer(r1, r2, h, b)
end

# all fields are numerical parameters
Functors.@functor SylvesterLayer

function construct_orthogonal(House::AbstractMatrix{<:Real})
    dims = size(House, 1)
    # normalize cols 
    H = House ./ norm.(eachcol(House))'
    # construct Q mat by multiply householder mats
    Q = I(dims)
    for h in eachcol(H)
        Q *= I(dims) .- 2.0*h*h'
    end
    return Q
end

# ensure r1*r1 > -1
function diag_constraint(r::AbstractMatrix{<:Real})
    r_diag = Diagonal(r)
    # rn_diag =  Diagonal(LogExpFunctions.log1pexp.(-diag(r_diag)) .- 1.0)
    rn_diag =  abs.(r_diag)
    rn = r .- r_diag .+ rn_diag
    return UpperTriangular(rn), rn
end

function _transform(flow::SylvesterLayer, z::AbstractVecOrMat{<:Real})  
    Q = construct_orthogonal(flow.h)
    r11, r11_diag = diag_constraint(flow.r1)
    Qr = Q*r11
    r22, r22_diag =  diag_constraint(flow.r2) 
    rQ = r22*Q  
    b = flow.b
    z1 = tanh.(rQ*z .+ b)
    transformed = z + Qr*z1
    rr_diag = r11_diag.*r22_diag
    return (transformed = transformed, rr_diag = rr_diag, z1 = z1)
end

(b::SylvesterLayer)(z) = Bijectors._transform(b, z).transformed

function Bijectors.with_logabsdet_jacobian(flow::SylvesterLayer, z::AbstractVecOrMat{<:Real})
    transformed, rr_diag, z1 = _transform(flow, z)
    log_det_jac = sum(log, abs.( 1.0 .+ (1.0 .- z1.^2.0).*rr_diag))
    return (transformed, log_det_jac)
end

Bijectors.logabsdetjac(flow::SylvesterLayer, x::AbstractVector{<:Real}) = last(Bijectors.with_logabsdet_jacobian(flow, x))


