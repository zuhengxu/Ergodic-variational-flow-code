using Distributions, ForwardDiff, LinearAlgebra, Random, Plots, ProgressMeter
using Base.Threads:@threads
using Zygote:@adjoint
using JLD
using Revise, ErgFlow
include("../../inference/SVI/svi.jl")

d = 2

function logp(z)
    g1 = -0.5*z[1]^2/.15^2 - 0.5*(z[2]-2)^2/1^2 - log(2π) - log(0.15)
    g2 = -0.5*(z[1] + 2.0)^2/1^2 - 0.5*z[2]^2/.15^2 - log(2π) - log(0.15)
    g3 = -0.5*z[2]^2/.15^2 - 0.5*(z[1]-2)^2/1^2 - log(2π) - log(0.15)
    g4 = -0.5*(z[2] + 2.0)^2/1^2 - 0.5*z[1]^2/.15^2 - log(2π) - log(0.15)
    return ErgFlow.logsumexp([g1, g2, g3, g4]) - log(4)
end

∇logp(x) = ForwardDiff.gradient(logp, x)
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))


#### contour of MF gaussian fit
# create the figure folder
### fit MF Gaussian
fig_dir =  "example/cross/figure"
res_dir =  "example/cross/result"

o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 1000)

x = -5.:.1:5
y = -5:.1:5
f = (x,y) -> exp(logp([x, y]))
gsvi = (x, y) -> exp(logq([x, y], μ, D))
# p1 = contour(x, y, f, colorbar = false, title = "Gaussian mixture")
# p2= contour(x, y, gsvi, colorbar = false, title = "MF Gaussian fit")
# pp = plot(p1, p2, p3, layout = 3)
# savefig(pp, joinpath(fig_dir,"contour.png"))


if ! isdir("figure")
    mkdir("figure")
end 
if ! isdir("result")
    mkdir("result")
end 