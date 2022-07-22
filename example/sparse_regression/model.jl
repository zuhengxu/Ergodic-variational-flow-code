using Distributions, ForwardDiff, LinearAlgebra, Random, Plots
using MLDatasets, Tullio
using Base.Threads:@threads
using JLD, DelimitedFiles
using Zygote:Buffer, ignore, gradient, @adjoint, @ignore
include("../../inference/ErgFlow/ergodic_flow.jl")
include("../../inference/SVI/svi.jl")

# read and standarize prostate_cancer data
prostate_dat = readdlm("data/pros_dat.txt")[2:end, 2:end]
X_raw = prostate_dat[:, 1:end-1]
Y_raw = prostate_dat[:, end]
# standarize dataset 
X = (X_raw .- mean(X_raw, dims =1)) ./ std(X_raw, dims = 1)
Y = (Y_raw .- mean(Y_raw))./std(Y_raw)
xs = hcat(X, Y) # 97 obs, 8 features, 1 response
N = size(xs, 1)
p = size(xs,2) - 1 # number of features
fs = hcat(ones(N), xs[:,1:p])
rs = @view(xs[:, p+1])
d = 2*(p+1) + 1 # 2 * size = betas + lambdas (including intercept); 1 = error variance
ϵ0 = 0.02 .* ones(d)
τ = 0.1
# z = [logσ²,β₁,...,βₚ,β_{p+1},λ₁,...,λₚ,λ_{p+1}]
function log_prior(z)
    z3 = @view(z[p+3:end]) .+ 1e-8
    # logσ²
    logp = -0.5 * (z[1]^2) - 0.5 * log(2*pi)
    # β₁,...,βₚ,β_{p+1}
    logp += -0.5 * sum( @view(z[2:p+2]).^2 ./ (τ^2 .* z3.^2)) - 0.5 * (p+1) * log(2π) - (p+1) * log(τ) - sum(log, abs.(z3))
    # λ₁,...,λₚ,λ_{p+1} 
    logp += -(p+1) * log(pi) - sum(log1p, z3.^2)
    return logp
end

function logp_lik(z)
    z1 =z[1]
    diffs = rs .- fs * @view(z[2:p+2])
    return -0.5 * exp(-z1) * sum(abs2, diffs) - 0.5 * N * (log(2π) + z1)
end
function ∇potential_by_hand(z)
    z2 = @view(z[2:p+2])
    z3 = @view(z[p+3:end]) .+ 1e-8
    z1 = z[1]
    diffs = rs .- fs * z2

    grads_p = zeros(d)
    grads_p[1] = -z1 + 0.5 * sum(abs2, diffs) * exp(-z1) - N/2
    @tullio s[j] :=  diffs[i]*fs[i,j]
    grads_p[2:p+2] .= -z2 ./ (τ^2 .* z3.^2) .+ s./exp(z1) 
    grads_p[p+3:d] .= -1 ./ z3 .+ z2.^2 ./ τ^2 ./ z3.^3.0 .- 2.0 .* z3 ./ (1 .+ z3.^2.0)

    return grads_p
end
∇logp(z) = @ignore @inbounds(∇potential_by_hand(z))

# ∇logp(z) = @ignore ForwardDiff.gradient(logp, z)
function logp(z)
    return log_prior(z) + logp_lik(z)
end
# Customize zygote gardient
@adjoint logp(z) = logp(z), Δ -> (∇logp(z), )

logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)
