using Flux, Zygote, JLD, Plots
using Revise, ErgFlow
include("model.jl")
include("../../inference/HVI/HVI.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")

# ### fit MF Gaussian
# Random.seed!(1)
# o1 = SVI.MFGauss(d, logp, randn, logq)
# el_svi = SVI.ELBO(o1, zeros(d), 0.01*ones(d); elbo_size = 10000)

###########
## EF
###########
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
Random.seed!(1)
ELBO_plot(o, o1; μ=zeros(d), D = 0.01*ones(d), eps = [1e-4, 2e-3, 8e-3], Ns = [100, 200, 500, 1000, 1500], nBs = [0,0,0,0,0,0], elbo_size = 5000, 
        res_name = "el5.jld",fig_name = "lg_elbo5.png", title = "Logistic regression", 
        xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

Random.seed!(1)
ksd_plot(o; μ = zeros(d), D = 0.01*ones(d), ϵ = 2e-3*ones(d), Ns = [100, 200, 500, 1000, 1500], nBs = [0], nsample = 5000, title  = "Logistic regression")

