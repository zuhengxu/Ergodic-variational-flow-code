using Distributions, ForwardDiff, LinearAlgebra, Random, Plots
using MLDatasets
using Base.Threads:@threads
using Base.Threads
using JLD, Tullio
using Zygote, ErgFlow
using Zygote:Buffer, ignore, gradient, @ignore
# using ChainRules
include("../../inference/SVI/svi.jl")

using DataFrames, CSV
# 506 instances, 13 features, 1 response
X_raw= Matrix(BostonHousing().features)
Y_raw= Matrix(BostonHousing().targets)
# X_raw= Matrix(BostonHousing.features()')
# Y_raw= Matrix(BostonHousing.targets()')
# standarize dataset 
X = (X_raw .- mean(X_raw, dims =1)) ./ std(X_raw, dims = 1)
Y = (Y_raw .- mean(Y_raw))./std(Y_raw)

# df = DataFrame(xs, :auto)
# CSV.write("result/Dat.csv", df)
xs = hcat(X, Y)
N = size(xs, 1)
d = size(xs, 2) + 1 # 13 features plus intercept and error variance
fs = hcat(ones(N), xs[:,1:d-2])
rs = xs[:, d-1]
ϵ0= 0.02 .* ones(d)

# standard normal (prior)
function log_prior(z)
    -0.5 * dot(z, z) - 0.5*d* log(2*pi)
end

function logp_lik(β, logσ)
    diffs = rs .- fs * β
    return -0.5 * exp(-logσ) * sum(abs2, diffs) - 0.5 * N * log(2π) - 0.5 * N * logσ
end

function ∇logp(z)
    β = @view(z[1:d-1])
    logσ = z[d]
    diffs = rs .- fs * β
    @tullio s[j] :=  diffs[i]*fs[i,j]
    gβ = -β+ exp(-logσ) .* s
    # vec(sum((diffs ./ exp(z[1])) .* fs, dims=1))
    gs = -logσ - N/2 + 0.5 * exp(-logσ) * sum(abs2, diffs)
    return vcat(gβ, [gs])
end

function logp(z)
    β = @view(z[1:d-1])
    logσ = z[d]
    return log_prior(z) + logp_lik(β, logσ)
end


logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)



folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 