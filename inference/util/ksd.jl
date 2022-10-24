######################
# ksd
######################
using Base.Threads, Distances, LinearAlgebra, ProgressMeter

# function imq_kernel(x, y, c, β)
#     return (c^2 + norm(x-y)^2)^β
# end

# function ∇x(x, y, c, β)
#     return 2 * β * ((c^2 + norm(x-y)^2)^(β-1)) * (x-y)
# end

# function ∇y(x, y, c, β)
#     return 2 * β * ((c^2 + norm(x-y)^2)^(β-1)) * (y-x)
# end

# function trace_term(x, y, c, β)
#     k = c^2 + norm(x-y)^2
#     d = size(x,1)
#     return -2 * β * d * (k^(β-1)) - 4 * β * (β-1) * (k^(β-2)) * norm(x-y)^2
# end

# function Up(x, y, c, β, grd)
#     return imq_kernel(x, y, c, β) * dot(grd(x), grd(y)) + dot(grd(x), ∇y(x, y, c, β)) + dot(grd(y), ∇x(x, y, c, β)) + trace_term(x, y, c, β)
# end

function k0(x::AbstractVector{Float64}, 
            y::AbstractVector{Float64}, 
            grd::Function; 
            c2 = 1.0, β= 0.5)
    # compute all the core values only once and store them
    d = length(x)

    z = x - y
    r2 = sum(abs2, z)
    base = c2 + r2
    base_beta = base^(-β)
    base_beta1 = base_beta / base

    gradlogpx, gradlogpy = grd(x), grd(y)

    coeffk = dot(gradlogpx, gradlogpy)
    coeffgrad = -2.0 * β * base_beta1

    kterm = coeffk * base_beta
    gradandgradgradterms = coeffgrad * (
        (dot(gradlogpy, z) - dot(gradlogpx, z)) +
        (-d + 2 * (β + 1) * r2 / base)
    )
    kterm + gradandgradgradterms
end

function ksd(D, grd::Function; c2 = 1.0, β = 0.5)
    #=
    D: sample matrix (each row is a data point)
    grd: ∇logp ---score of target distribution
    =#
    N = size(D,1)
    ksd = Threads.Atomic{Float64}(0.0)
    @info "computing KSD" 
    prog_bar = ProgressMeter.Progress(N^2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    @threads for i in 1:N 
        @simd for j in 1:N
            @inbounds kij = k0(@view(D[i,:]), @view(D[j,:]), grd; c2 = c2, β = β)
            Threads.atomic_add!(ksd, kij)
            ProgressMeter.next!(prog_bar)
        end
    end
    return sqrt(ksd[])/N
end

