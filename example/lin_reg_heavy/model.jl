using Distributions, LinearAlgebra, Random
using ErgFlow
using Tullio, ForwardDiff
using Base.Threads:@threads
using JLD, DataFrames, CSV 
using Zygote:Buffer, ignore, gradient, @ignore, @adjoint
include("../../inference/SVI/svi.jl")


function data_load(dat)
```
load and process .npz data
```
    Dat = CSV.read(dat, DataFrame; header = 0)
    # turn String31 into Float64
    for c in names(Dat)
        if eltype(Dat[!,c]) != Float64
            Dat[!, c] = parse.(Float64, Dat[!, c])
        end
    end
    Zr, Y = Dat[:, 1:end-1], Dat[:,end]
    N, p = size(Zr)
    Z = hcat(ones(N), Zr)
    return Matrix(Z), Y, N, p
end

############
# read processed Communities and Crime dataset : http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
############
cd("/arc/project/st-tdjc-1/mixflow/Ergodic-variational-flow-code/example/lin_reg_heavy")
fs, rs, N, p = data_load("data/communities_pca.csv")
d = p+2

#######################
# models ( lin reg with heavy tailed prior, β ∼ Cauchy(0, 1))
#######################
# s := logσ²
@inline function log_pr_s(s) 
    -0.5 * (s^2) - 0.5 * log(2*pi)
end

function log_pr_β(β)
    return -d*log(π) - sum(log1p, β.^2.0)
end

#  β:= β₁,...,βₚ,β_{p+1}
function log_lik(s, β)
    diffs = rs .- fs * β
    return -0.5 * exp(-s) * sum(abs2, diffs) - 0.5 * N * (log(2π) + s)
end

# z = (logσ², β)
function logp(z)
    s = z[1]
    β = @view(z[2:end])
    return log_pr_s(s) + log_pr_β(β) + log_lik(s, β)
end


@inline function ∇log_pr_β(β::Real)
    return -2.0*β/(1.0 + β^2.0)
end

function ∇logp(z)
    s = z[1]
    β = @view(z[2:end])
    diffs = rs .- fs * β
    @tullio g[j] :=  diffs[i]*fs[i,j]
    gβ = exp(-s) .* g .+ ∇log_pr_β.(β)
    # vec(sum((diffs ./ exp(z[1])) .* fs, dims=1))
    gs = -s - N/2 + 0.5 * exp(-s) * sum(abs2, diffs)
    return vcat([gs], gβ)
end


# using BenchmarkTools
# G(x) = ForwardDiff.gradient(logp, x)
# at = randn(d) 
# @btime ∇logp(at)
# @btime G(at)

# # Customize zygote gardient
# @adjoint logp(z) = logp(z), Δ -> (Δ*∇logp(z), )

########################
# SVI
########################
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)

# if ! isdir("figure")
#     mkdir("figure")
# end 
# if ! isdir("result")
#     mkdir("result")
# end 