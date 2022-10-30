include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
# include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs

Random.seed!(1)
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 

Random.seed!(1)
ELBO_plot(o, o1; μ=μ, D=D, eps = [0.005, 0.01, 0.015], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0], elbo_size = 1000, 
        fig_name = "2d_funnel_elbo.png",title = "Neal's Funnel", xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
els = eps_tunning([0.001:0.002:0.025 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000,fig_name = "funnel_tune.png", title = "Neal's Funnel", 
                xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18,xrotation = 20)

Random.seed!(1)
ksd_plot(o; μ = μ, D = D, ϵ = 0.012*ones(2), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =2000, title  = "Neal's Funnel")

################3
## contour and scatter
################
Random.seed!(1)
x = -20:0.1:30
y = -30:0.1:30
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
scatter_plot(o, x, y; contour_plot = true, μ=μ, D=D, ϵ = 0.012*ones(d), n_sample = 1000, n_mcmc = 500, nB = 0, bins = 500, name= "2d_funnel_sample.png")

#####################
# lpdf_est
####################
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std,ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
a = ErgFlow.HF_params(0.012*ones(d), μ, D)

X = [-30.001:0.5:30 ;]
Y = [-30.001:0.5:30 ;]
# lpdf_est, lpdf, Error
DS, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 50)

# # #####################
# # # lpdf_est
# # ####################
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std,ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
a = ErgFlow.HF_params(0.01*ones(d), μ, D)

X = [-20.001:0.5:20 ;]
Y = [-15.001:0.5:20 ;]
# lpdf_est, lpdf, Error
DS, Dd, E = lpdf_est_save(o, a, X, Y; n_mcmc = 1000, nB = 50)


layout = pjs.Layout(
    width=500, height=500,
    scene = pjs.attr(
        xaxis = pjs.attr(showticklabels=false, visible=false),
        yaxis = pjs.attr(showticklabels=false, visible=false),
        zaxis = pjs.attr(showticklabels=false, visible=false, range = [-5000, 0]),
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

# p_est = pjs.plot(pjs.surface(z=DS, x=X, y=Y, cauto = false, cmax = 0, cmin = -5000, showscale=false), layout)
# pjs.savefig(p_est, joinpath("figure/","lpdf_est.png"))