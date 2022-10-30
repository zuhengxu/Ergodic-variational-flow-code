using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, CSV, DataFrames
using Base.Threads, ErgFlow
using Base.Threads:@threads
using JLD, Tullio, ProgressMeter
using Zygote
using Zygote:Buffer, ignore, gradient, @adjoint, @ignore
include("../../inference/SVI/svi.jl")
#########################################################################################
# Example of Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
# The observed data D = {X, y} consist of N binary class labels, 
# y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
# The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
# and a precision parameter \alpha \in R_+. We assume the following model:
#     p(α) = Gamma(α ; a, b) , τ = log α ∈ R  
#     p(w_k | τ) = N(w_k; 0, exp(-τ))
#     p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t)), y ∈ {1, 0}
#########################################################################################
###########
## load dataset 
###########
# dat = load("data/dataset.jld")
# X, Y = dat["X"], dat["Y"]

df = DataFrame(CSV.File("data/final_dat.csv"))
xs = Matrix(df)[:, 2:end]
N = size(xs, 1)
X = xs[:, 1:end-1]
X = (X .- mean(X, dims = 1))./std(X, dims=1)
Y = xs[:, end] 

p = size(X, 2)
d = p+1
a, b = 1.0, 0.01
##########
# log posterior
##########
function log_sigmoid(x)
    if x < -300
        return x
    else
        return -log1p(exp(-x))
    end
end

function neg_sigmoid(x)
    return -1.0/(1.0 + exp(-x))
end

# z = (τ, w1, ..., wd)
function logp(θ)
    τ = θ[1]
    W = @view(θ[2:end])
    Z = X*W
    logpτ = a*τ - b*exp(τ)
    logpW = 0.5*p*τ - 0.5*exp(τ)* sum(abs2, W)
    @tullio llh := (Y[n] -1.0) *Z[n] + log_sigmoid(Z[n])
    # llh = sum((Y .- 1.) .* Z .- log1p.(exp.(-Z)))
    return logpτ + logpW  + llh
end

function ∇logp(z)
    τ = z[1]
    W = @view(z[2:end])
    grad = similar(z)
    grad[1] = a - b*exp(τ) + 0.5*p - 0.5*exp(τ)* sum(abs2, W)
    S = neg_sigmoid.(X*W) 
    # @tullio S[n] := X[n,j]*W[j] |> neg_sigmoid  
    @tullio M[j] := X[n,j]*(S[n] + Y[n])
    grad[2:end] .= -exp(τ).*W .+ M
    return grad
end


# customize gradient for logp
Zygote.refresh()
@adjoint logp(z) = logp(z), Δ -> (Δ * ∇logp(z), )

logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)




if ! isdir("figure")
    mkdir("figure")
end 
if ! isdir("result")
    mkdir("result")
end 