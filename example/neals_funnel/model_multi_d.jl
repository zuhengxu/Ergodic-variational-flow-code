using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
include("../../inference/ErgFlow/ergodic_flow.jl")
include("../../inference/SVI/svi.jl")

# d = 5 # dimension (>=2) σ² = 9 # variance of first dimension
# σ² = 36
# Z = sqrt(σ² * (2*π)^d)*exp((d-1)^2*σ²/32) # normalizing constant
# ## x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) <- missing cross term
# logp(x) = -log(Z) - 0.5 * x[1]^2/σ² - 0.5 * sum(x[2:end]' * x[2:end]) / exp(0.5*x[1])
# ## x1∼N(0, σ²), x2∼N(0, exp(x1/2)) 
# ∇logp(x) = vcat(-x[1]/σ² + 0.25 * x[2:end]' * x[2:end] * exp(-x[1]/2), -x[2:end] .* exp(-x[1]/2))

# # funnel that contains the cross term (first dimension shifted)
# Z = sqrt(σ² * (2*π)^d) # normalizing constant
# # x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) 
# logp(x) = -log(Z) - 0.5 * (x[1]-σ²/4)^2/σ² - 0.5 * (x[2:end]' * x[2:end]) / exp(0.5*x[1]) - (d-1)*x[1]/4
# ∇logp(x) = vcat(-(x[1]-σ²/4)/σ² + 0.25 * x[2:end]' * x[2:end] * exp(-x[1]/2) - (d-1)/4, -x[2:end] .* exp(-x[1]/2))

logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./D)