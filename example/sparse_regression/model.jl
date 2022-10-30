using Distributions, LinearAlgebra, Random, Plots
using MLDatasets, Tullio, ErgFlow
using Base.Threads:@threads
using JLD, DelimitedFiles
using Zygote:Buffer, ignore, gradient, @ignore, @adjoint
include("../../inference/SVI/svi.jl")


function covariates_std(X_raw)
    # extract covariates for stdization
    N, p= size(X_raw)
    X = (X_raw .- mean(X_raw, dims =1)) ./ std(X_raw, dims = 1) 
    Z = hcat(ones(N), X) 
    return Z, N, p
end
############
# read and standarize prostate_cancer data
############
prostate_dat = readdlm("data/pros_dat.txt")[2:end, 2:end]
X_raw = prostate_dat[:, 1:end-1]
Y_raw = prostate_dat[:, end]
# Y = (Y_raw .- mean(Y_raw))./std(Y_raw)
# standarize dataset 
fs, N, p = covariates_std(X_raw)
rs =  Vector{Float64}(Y_raw)
# xs = hcat(fs, Y) # 97 obs, 8 features, 1 response
d = p+2
# ϵ0 = 0.02 .* ones(d)
τ = 0.05


################
# model 
################

# s := logσ²
@inline function log_pr_s(s) 
    -0.5 * (s^2) - 0.5 * log(2*pi)
end
function log_pr_β(β)
    T = β.^2.0.*[-50.0 -0.005] .- [log(0.1) log(10)]
    return sum(logsumexp(T; dims =2)) - (p+1)*(0.5*log(2π) +log(2.0))  
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

if ! isdir("figure")
    mkdir("figure")
end 
if ! isdir("result")
    mkdir("result")
end 