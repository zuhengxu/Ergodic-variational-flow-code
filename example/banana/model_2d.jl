using Distributions, ForwardDiff, Random, Plots, ProgressMeter, LinearAlgebra, Random
using Base.Threads:@threads
using JLD
using ErgFlow



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




if ! isdir("figure")
    mkdir("figure")
end 
if ! isdir("result")
    mkdir("result")
end 