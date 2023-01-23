using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
using JLD
using ErgFlow

d = 2 # dimension (>=2) σ² = 9 # variance of first dimension
σ² = 36
Z = sqrt(σ² * (2*π)^d) # normalizing constant
## x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) # not quite --> pdf of x1 is missing the cross term
logp(x) = -log(Z) -σ²/32 - 0.5 * x[1]^2/σ² -0.5*exp(-0.5*x[1])*x[2]^2.0
## x1∼N(0, σ²), x2∼N(0, exp(x1/2)) 
# logp(x) = -log(Z) -0.25*x[1]- 0.5 * x[1]^2/σ² -0.5*exp(-0.5*x[1])*x[2]^2.0
∇logp(x) = vcat(-x[1]/σ² + x[2]^2.0 * 0.25 * exp(-x[1]/2), -x[2]*exp(-x[1]/2))
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./D)


folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 