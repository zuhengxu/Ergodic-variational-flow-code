include("model_2d.jl")
include("../../inference/SVI/svi.jl")
using JLD
using Base.Threads: @threads 
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs
Random.seed!(1)

##################33
# setting
#####################
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
o1 = SVI.MFGauss(d, logp, randn, logq)

#########################3
#  elbo
#########################3
Random.seed!(1)
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.001, 0.003, 0.006], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0,0], elbo_size = 2000, 
title = "Warped Gaussian", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20, 
fig_name = "elbo_lap.png", res_name = "elbo_lap.jld")

# Random.seed!(1)
# els = eps_tunning([0.001:0.0005:0.005 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000, fig_name = "warp_tune.png", title = "Warped Gaussian",
#                  xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

#########################3
#  ksd
#########################3
Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.003*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =5000, title  = "Warped Gaussian", fig_name = "ksd_lap.png", res_name = "ksd_lap.jld")

# ################3
# contour and scatter
# ################
Random.seed!(1)
x = -2:.01:2
y = -4:.01:4
scatter_plot(o, x, y; contour_plot = false, μ=μ, D=D, ϵ = 0.003*ones(d), n_sample = 1000, n_mcmc = 1000, nB = 0, bins = 500, name= "scatter_lap.png")


#####################
# lpdf_est
####################
a = ErgFlow.HF_params(0.003*ones(d), μ, D)

X = [-2.01:.1:2 ;]
Y = [-5.01:.1:5 ;]
# # lpdf_est, lpdf, Error
DS, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 0)

layout = pjs.Layout(
    width=500, height=500,
    scene = pjs.attr(
    xaxis = pjs.attr(showticklabels=true, visible=true),
    yaxis = pjs.attr(showticklabels=true, visible=true),
    zaxis = pjs.attr(showticklabels=true, visible=true),
    ),
    margin=pjs.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)


p_target = pjs.plot(pjs.surface(z=Dd, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_target, joinpath("figure/","lpdf.png"))

p_est = pjs.plot(pjs.surface(z=DS, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_est, joinpath("figure/","lpdf_lap.png"))

