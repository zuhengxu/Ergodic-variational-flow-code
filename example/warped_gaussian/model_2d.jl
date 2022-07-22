using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
using JLD
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

################# 
#fit MF Gaussian
################## 
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 10000)
