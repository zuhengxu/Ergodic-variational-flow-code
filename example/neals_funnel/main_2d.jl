include("model_2d.jl")
# include("../../inference/EF/ErgFlow.jl")
include("../../inference/SVI/svi.jl")
using JLD
using Base.Threads: @threads 
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs


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
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.005, 0.012, 0.015], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0,0], elbo_size = 2000, 
title = "Neal's Funnel", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20, 
fig_name = "elbo_lap.png", res_name = "elbo_lap.jld")

#########################3
#  ksd
#########################3
Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.012*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =5000, title  = "Neal's Funnel", fig_name = "ksd_lap.png", res_name = "ksd_lap.jld")

################3
## contour and scatter
################
Random.seed!(1)
x = -20:0.1:30
y = -30:0.1:30
scatter_plot(o, x, y; contour_plot = false, μ=μ, D=D, ϵ = 0.012*ones(d), n_sample = 1000, n_mcmc = 500, nB = 0, bins = 500, name= "scatter_lap.png")

#####################
# lpdf_est
####################
a = ErgFlow.HF_params(0.012*ones(d), μ, D)

X = [-30.001:0.5:30 ;]
Y = [-30.001:0.5:30 ;]

# X = [-20.001:0.5:20 ;]
# Y = [-15.001:0.5:20 ;]
# lpdf_est, lpdf, Error
DS, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 2000, nB = 5)


layout = pjs.Layout(
    width=500, height=500,
    scene = pjs.attr(
        xaxis = pjs.attr(showticklabels=true, visible=true),
        yaxis = pjs.attr(showticklabels=true, visible=true),
        zaxis = pjs.attr(showticklabels=true, visible=true, range = [-5000, 0]),
    ),
    margin=pjs.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)

keep = Dd .> -6000
new_Dd = zeros(121, 121)
for i in 1:121
    for j in 1:121
        if keep[i,j] == 1
            new_Dd[i,j] = Dd[i,j]
        else
            new_Dd[i,j] = NaN
        end
    end
end

p_target = pjs.plot(pjs.surface(z=Dd, x=X, y=Y, cauto = false, cmax = 0, cmin = -5000, showscale=false), layout)
pjs.savefig(p_target, joinpath("figure/","lpdf.png"))

p_est = pjs.plot(pjs.surface(z=DS, x=X, y=Y, cauto = false, cmax = 0, cmin = -5000, showscale=false), layout)
pjs.savefig(p_est, joinpath("figure/","lpdf_lap.png"))