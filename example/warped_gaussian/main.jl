 include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs

Random.seed!(1)
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 

Random.seed!(1)
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.001, 0.0023, 0.006], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0,0], elbo_size = 1000, 
        fig_name = "2d_warp_elbo.png",title = "Warped Gaussian", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)
Random.seed!(1)
els = eps_tunning([0.001:0.0005:0.005 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000, fig_name = "warp_tune.png", title = "Warped Gaussian",
                 xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.003*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =2000, title  = "Warped Gaussian")
# ################3
# ## contour and scatter
# ################
Random.seed!(1)
x = -2:.01:2
y = -4:.01:4
scatter(o, x, y; contour_plot = true, μ=μ, D=D, ϵ = 0.005*ones(d), n_sample = 1000, n_mcmc = 500, nB = 50, bins = 500, name= "2d_warp_sample.png")


#####################
# lpdf_est
####################
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std,ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
a = ErgFlow.HF_params(0.003*ones(d), μ, D)

X = [-2.01:.1:2 ;]
Y = [-5.01:.1:5 ;]
# # lpdf_est, lpdf, Error
DS, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 50)

layout = PlotlyJS.Layout(
    width=500, height=500,
    scene = PlotlyJS.attr(
    xaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    yaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    zaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    ),
    margin=PlotlyJS.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)

p_target = pjs.plot(pjs.surface(z=Dd, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_target, joinpath("figure/","lpdf.png"))

p_est = pjs.plot(pjs.surface(z=DS, x=X, y=Y, showscale=false), layout)
pjs.savefig(p_est, joinpath("figure/","lpdf_est.png"))

