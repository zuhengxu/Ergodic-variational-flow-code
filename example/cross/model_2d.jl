using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter, LogExpFunctions
using Base.Threads:@threads
using Zygote:@adjoint
using ErgFlow

d = 2

function logp(z)
    g1 = -0.5*z[1]^2/.15^2 - 0.5*(z[2]-2)^2/1^2 - log(2π) - log(0.15)
    g2 = -0.5*(z[1] + 2.0)^2/1^2 - 0.5*z[2]^2/.15^2 - log(2π) - log(0.15)
    g3 = -0.5*z[2]^2/.15^2 - 0.5*(z[1]-2)^2/1^2 - log(2π) - log(0.15)
    g4 = -0.5*(z[2] + 2.0)^2/1^2 - 0.5*z[1]^2/.15^2 - log(2π) - log(0.15)
    return LogExpFunctions.logsumexp([g1, g2, g3, g4]) - log(4)
end

∇logp(x) = ForwardDiff.gradient(logp, x)
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))

folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 