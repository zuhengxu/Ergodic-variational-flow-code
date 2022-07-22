using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
using JLD
include("../../inference/SVI/svi.jl")


d = 2 # dimension (>=2) σ² = 9 # variance of first dimension
σ² = 36
Z = sqrt(σ² * (2*π)^d) # normalizing constant
## x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) # not quite --> pdf of x1 is missing the cross term
logp(x) = -log(Z) -σ²/32 - 0.5 * x[1]^2/σ² -0.5*exp(-0.5*x[1])*x[2]^2.0
## x1∼N(0, σ²), x2∼N(0, exp(x1/2)) 
# logp(x) = -log(Z) -0.25*x[1]- 0.5 * x[1]^2/σ² -0.5*exp(-0.5*x[1])*x[2]^2.0
∇logp(x) = vcat(-x[1]/σ² + x[2]^2.0 * 0.25 * exp(-x[1]/2), -x[2]*exp(-x[1]/2))
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./D)


#### contour of MF gaussian fit
# create the figure folder
### fit MF Gaussian
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 10000)


folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 