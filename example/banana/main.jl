include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
# include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs
Random.seed!(1)

###########3
# ELBO/KSD
###########
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

ELBO_plot(o, o1; μ=μ, D=D, eps = [0.005, 0.013, 0.02], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0, 0, 0, 10, 20, 20], elbo_size = 1000, 
fig_name = "2d_banana_elbo.png", title = "Banana", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
els = eps_tunning([0.008:0.002:0.03 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000, fig_name = "banana_tune.png", title = "Banana", 
                kxtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18)

Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.016*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =2000, title  = "Banana")

# ErgFlow.ELBO(o, 0.01*ones(o.d), μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord, 1000; nBurn = 0, elbo_size = 1000, print = true)
# ErgFlow.ELBO_long(o, 0.01*ones(o.d), μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord, 1000; nBurn = 0, elbo_size = 1000, print = true)

####################
#### sctter plot 
####################
Random.seed!(1)
x = -20:0.1:20
y = -15:0.1:30
scatter_plot(o, x, y; contour_plot = true, μ=μ, D=D, ϵ = 0.02*ones(d), n_sample = 1000, n_mcmc = 500, nB = 0, bins = 500, name= "2d_banana_sample.png", show_legend=true)


################
# lpdf estimation 
###############
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
a = ErgFlow.HF_params(0.016*ones(d), μ, D)
X = [-20.001:0.5:20 ;]
Y = [-15.001:0.5:20 ;]

# lpdf_est, lpdf, Error
Ds, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 20)

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
pjs.savefig(p_est, joinpath("figure/","lpdf_est.png"))
