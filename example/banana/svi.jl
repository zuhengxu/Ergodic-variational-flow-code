using JLD
include("model_2d.jl")
include("../../inference/SVI/svi.jl")
using .SVI

#### contour of MF gaussian fit
# create the figure folder
### fit MF Gaussian
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 10, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 1000)
JLD.save("result/mfvi.jld", "μ", μ, "D", D, "elbo", el_svi)

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
x = -20.:.1:20
y = -15:.1:30
f = (x,y) -> exp(logp([x, y]))
gsvi = (x, y) -> exp(logq([x, y], μ, D))
p1 = contour(x, y, f, colorbar = false, title = "Gaussian mixture")
p2 = contour(x, y, gsvi, colorbar = false, title = "MF Gaussian fit")
pp = plot(p1, p2, layout = 2)
savefig(pp, joinpath("figure/","contour.png"))