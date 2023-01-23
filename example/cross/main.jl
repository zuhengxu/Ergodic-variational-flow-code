include("model_2d.jl")
include("../../inference/SVI/svi.jl")
using JLD
using Base.Threads: @threads 
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs


###########3
#  setting
###########
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
n_lfrg = 60
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
o1 = SVI.MFGauss(d, logp, randn, logq)

###########3
# ELBO
###########
Random.seed!(1)
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.001, 0.0035, 0.008], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0, 0, 0, 0, 0, 0], elbo_size = 2000, 
title = "Cross", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20,
res_name = "elbo_lap.jld", fig_name = "elbo_lap.png")

Random.seed!(1)
els = eps_tunning([0.002:0.0005:0.01 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000,
                fig_name = "cross_tune.png", title = "Cross", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.0035*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =5000, title  = "Cross", fig_name="ksd_lap.png", res_name = "ksd_lap.jld")

####################
#### sctter plot u
####################
Random.seed!(1)
x = -5:0.1:5
y = -5:0.1:5
scatter_plot(o, x, y; contour_plot = true, μ=μ, D=D, ϵ = 0.0035*ones(d), n_sample = 1000, n_mcmc = 500, nB = 0, bins = 500, name= "scatter_lap.png", show_legend=false)

# # ################
# # # lpdf estimation 
# # ###############
a = ErgFlow.HF_params(0.0035*ones(d), μ, D)
X = [-5.01:.1:5 ;]
Y = [-5.01:.1:5 ;]

# # lpdf_est, lpdf, Error
Ds, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 5)

layout = pjs.Layout(
    width=500, height=500,
    scene = pjs.attr(
    xaxis = pjs.attr(showticklabels=false, visible=false),
    yaxis = pjs.attr(showticklabels=false, visible=false),
    zaxis = pjs.attr(showticklabels=false, visible=false),
    ),
    margin=pjs.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)

p_target = pjs.plot(pjs.surface(z=Dd, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_target, joinpath("figure/","lpdf.png"))

p_est = pjs.plot(pjs.surface(z=Ds, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_est, joinpath("figure/","lpdf_lap.png"))
