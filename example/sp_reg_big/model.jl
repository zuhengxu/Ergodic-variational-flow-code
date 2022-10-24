using Distributions, LinearAlgebra, Random, Plots
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
    # turn down datasize to make2 post looks wierd
    Random.seed!(1)
    # idx = sample(1:N, 100, replace = false)
    return Matrix(Z[1:50, :]), Y[1:50], N, p
end

############
# read processed Communities and Crime dataset : http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime
############
# fs, rs, N, p = data_load("data/communities_st.csv")
fs, rs, N, p = data_load("data/super_st.csv")
d =  p+2

################
# model 
################

# s := logσ²
@inline function log_pr_s(s) 
    -0.5 * (s^2) - 0.5 * log(2*pi)
end
function log_pr_β(β)
    T = β.^2.0.*[-50.0 -0.005] .- [log(0.1) log(10)]
    return sum(Flux.logsumexp(T; dims =2)) - (p+1)*(0.5*log(2π) +log(2.0))  
end
#  β:= β₁,...,βₚ,β_{p+1}
function log_lik(s, β)
    diffs = rs .- fs * β
    return -0.5 * exp(-s) * sum(abs2, diffs) - 0.5 * N * (log(2π) + s)
end
log_prior(s, β) = log_pr_β(β) + log_pr_s(s)

function logp(z)
    s = z[1] 
    β = @view(z[2:end])
    return log_prior(s, β) + log_lik(s, β)
end

# ∇logp(z) = ForwardDiff.gradient(logp, z)


function ∇log_pr_β1(β)
    b = exp(-(50.0 - 5e-3)*β^2.0)
    exponent = log1p(1e6*b) - 2.0*log(10.0) - log1p(1e2*b)
    return -β*exp(exponent)
end
function ∇log_pr_β(β)
    ∇log_pr_β1.(β)
end
@inline function ∇log_pr_s(s) 
    -s
end
function ∇log_lik(s, β)
    diffs = rs .- fs * β
    a = 0.5 * exp(-s)
    gs = a * sum(abs2, diffs) - 0.5 * N 
    gβ = fs' * diffs .* 2a
    return gs, gβ
end
function ∇logp(z)
    s = z[1] 
    β = @view(z[2:end])
    gsl, gbl = ∇log_lik(s, β) 
    gs = gsl + ∇log_pr_s(s)
    gb = gbl .+ ∇log_pr_β(β)
    return vcat([gs], gb)
end

# # Customize zygote gardient
# @adjoint logp(z) = logp(z), Δ -> (Δ*∇logp(z), )

########################
# SVI
########################
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)

