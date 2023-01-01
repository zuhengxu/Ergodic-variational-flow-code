using Flux, JLD, Plots
using ErgFlow
using Zygote
using Zygote: @adjoint
include("../../inference/SVI/svi.jl")
include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/ksd.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")


#####################
# ### fit MF Gaussian
#####################
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 10, logging_ps = false, optimizer = Flux.ADAM(1e-3))

μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000)
JLD.save(joinpath("result/", "mfvi.jld"), "μ", μ, "D", D, "elbo", el_svi)



###########
## ELBO
###########
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

Random.seed!(1)
ELBO_plot(o, o1; μ= μ, D = D, eps = [8e-5, 1e-4, 1.2e-4], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0,0,0,0,0,0,0,0], elbo_size = 1000, 
		res_name = "el.jld",fig_name = "poiss_elbo.png", title = "Poisson regression", 
		xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)


# ###########
# # KSD
# ###########
# Random.seed!(1)
# ksd_plot(o; μ = μ, D = D, ϵ = 1e-4*ones(d), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample = 5000, title  = "Poisson regression", fig_name = "ksd.png", res_name = "ksd.jld")
