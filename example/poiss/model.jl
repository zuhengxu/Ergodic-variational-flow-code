ENV["JULIA_PKG_PRECOMPILE_AUTO"]=0

using Tullio

using NPZ, LinearAlgebra, Distributions, Random
using ErgFlow
using LogExpFunctions
using Zygote
using Zygote:@adjoint, refresh

function data_load(dnm)
```
load and process .npz data
```
    dat = npzread(dnm)
    Zr = dat["X"]
    N, p = size(Zr)
    Y = dat["y"]
    # extract covariates for stdization
    Xr = Zr[:, 1:end-1] 
    chol = cholesky(Symmetric(cov(Xr) + 1e-8I))
    X = Matrix((chol.L\Xr')')
    Z = hcat(ones(N), X)
    return Z, Float64.(Y), N, p
end


### helper functions#####
function log_sigmoid_neg(x)
    # x> 300.0 ? x : log1p(exp(x))
    LogExpFunctions.log1pexp(x)
end
# function sigmoid(x)
#     return 1.0/(1.0 + exp(-x))
# end

#######################
# models
#######################

@inline function log_prior(x, d::Int)
    # logpτ = a*τ - b*exp(τ)
    # logpW = 0.5*p*τ - 0.5*exp(τ)* sum(abs2, W)
    return -0.5*d*log(2π) - 0.5* sum(abs2, x)
end

@inline function ∇log_prior(x)
    return -x
end

function log_lik(x, Z, Y)
    λ = log_sigmoid_neg.(Z*x) 
    @tullio llh := Y[n]*log(λ[n]) - λ[n]
    return llh
end

function ∇log_lik(x, Z, Y) 
    W = Z*x
    lsn = log_sigmoid_neg.(W)
    ls = LogExpFunctions.logistic.(W)
    @tullio ∇L[i] := (Y[n]/lsn[n] - 1.0) * ls[n]*Z[n, i]
    return ∇L 
end

function logp_joint(x, Z, Y, d::Int)
    return log_prior(x, d) + log_lik(x, Z, Y)
end


function ∇logp_joint(x, Z, Y)
    return ∇log_prior(x) .+ ∇log_lik(x, Z, Y)
end

####################3
# load dataset
#####################
cd("/arc/project/st-tdjc-1/mixflow/Ergodic-variational-flow-code/example/poiss")
Z, Y, N, d = data_load("data/airportdelays.npz")
logp(x::AbstractVector{Float64}) = logp_joint(x, Z, Y, d)
∇logp(x::AbstractVector{Float64}) = ∇logp_joint(x, Z, Y)

# # customize gradient for logp
# Zygote.refresh()
# @adjoint logp(z) = logp(z), Δ -> (Δ * ∇logp(z), )

##################
# SVI
##################
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)


# if ! isdir("figure")
#     mkdir("figure")
# end 
# if ! isdir("result")
#     mkdir("result")
# end 