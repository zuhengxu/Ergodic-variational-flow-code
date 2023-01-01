using Flux, Zygote, JLD, Plots
using ErgFlow
include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/ksd.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")

################
### fit MF Gaussian
##################
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)

a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 10000)
JLD.save("result/mf_params.jld", "μ", μ, "D", D, "elbo", el_svi)


MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
###########
##  ELBO
###########
n_lfrg =  50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

Random.seed!(1)
ELBO_plot(o, o1; μ=μ, D = D, eps = [8e-4, 1e-3, 1.2e-3], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0,0,0], elbo_size = 2000, 
        res_name = "el.jld",fig_name = "sp_elbo.png", title = "Sparse regression (high dim)", 
        xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)


# # ###########33
# # # KSD
# # ###############
# Random.seed!(1)
# ksd_plot(o; μ = μ, D = D, ϵ = 8e-4*ones(d), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample = 5000, title  = "Sparse regression (high dim)", fig_name = "ksd.png", res_name = "ksd.jld")