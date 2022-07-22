using Distributions, ForwardDiff, Random, Plots, ProgressMeter, LinearAlgebra
using Base.Threads:@threads
using JLD
using Revise, ErgFlow
# include("../../inference/ErgFlow/ergodic_flow.jl")
include("../../inference/SVI/svi.jl")


d = 2
b = 0.1 # curvature
Z = sqrt(100 * (2*π)^d) # normalizing constant
C = Matrix(Diagonal(vcat(100, ones(d-1))))
C_inv = Matrix(Diagonal(vcat(1/100, ones(d-1))))
ϕ_inv(y) = [y[1], y[2] - b*y[1]^2 + 100*b]
logp(x) = -log(Z) - 0.5 * ϕ_inv(x)' * C_inv * ϕ_inv(x)
∇logp(x) = -[1/100 * x[1] + (x[2]-b*x[1]^2+100*b)*(-2*b*x[1]), x[2]-b*x[1]^2+100*b]
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)


### fit MF Gaussian
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 1000)



folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 