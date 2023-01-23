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
n_lfrg = 200
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)


###########3
# ELBO/KSD
###########
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.005, 0.02, 0.03], Ns = [10, 20, 50, 80, 100, 120, 150, 200, 250, 300, 500], nBs = [0, 0, 0, 0, 0, 0, 0], elbo_size = 2000, 
fig_name = "elbo_lap.png", title = "Banana", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.02*ones(2), Ns = [10, 20, 50, 80, 100, 120, 150, 200, 250, 300], nBs = [0], nsample  = 5000, title  = "Banana", fig_name = "ksd_lap.png")

####################
#### sctter plot 
####################
Random.seed!(1)
x = -20:0.1:20
y = -15:0.1:30
scatter_plot(o, x, y; contour_plot = false, μ=μ, D=D, ϵ = 0.02*ones(d), n_sample = 1000, n_mcmc = 1000, nB = 0, bins = 300, name= "sample_lap.png", show_legend=false)


################
# lpdf estimation 
###############
X = [-20.001:0.5:20 ;]
Y = [-15.001:0.5:20 ;]

# lpdf_est, lpdf, Error
a = ErgFlow.HF_params(0.02*ones(2), μ, D)
Ds, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 500, nB = 5, res_name = "lpdf_lap.jld")

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
