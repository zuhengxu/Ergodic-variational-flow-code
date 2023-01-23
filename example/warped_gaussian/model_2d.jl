using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
using Zygote:@ignore
using JLD
using ErgFlow
# using Zygote: @ignore
include("../../inference/SVI/svi.jl")

function logp(z)
    x,y = z 
    r = sqrt(x^2+y^2)
    θ = atan(y, x) #in [-π , π]
    # increase θ depending on r to "smear"
    θ += r/2

    # get the x,y coordinates foαtransformed point
    x0 = r*cos(θ)
    y0 = r*sin(θ)
    logJ = log(r)
    
    # output the log density
    return -0.5*x0^2 - 0.5*y0^2/.12^2- log(2π) - log(0.12) + logJ
end
∇logp(x) = @ignore ForwardDiff.gradient(logp, x)
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./D)
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)
d = 2


folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 