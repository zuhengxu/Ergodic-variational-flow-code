using Flux, Zygote, JLD, Plots, ProgressMeter
include("model.jl")
include("../../inference/HVI/HVI.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")

# ### fit MF Gaussian
# Random.seed!(1)
# o1 = SVI.MFGauss(d, logp, randn, logq)
# a1 = SVI.mf_params(zeros(d), ones(d)) 
# ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false, optimizer = Flux.ADAM(1e-3))
# # Plots.plot(el1, ylims = (-50, 10))

# μ,D = ps1[1][1], ps1[1][2]
# el_svi = SVI.ELBO(o1, zeros(d), ones(d); elbo_size = 1000)


###########
##  ELBO
###########
n_lfrg = 30
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
Random.seed!(1)
ELBO_plot(o, o1; μ=zeros(d), D = 0.01*ones(d), eps = [0.0001, 0.0005, 0.001], Ns = [100, 200, 500, 1000,1500, 2000], nBs = [0,0,0,0,0,0], elbo_size = 2000, 
        res_name = "el.jld",fig_name = "lr_elbo.png", title = "Linear regression", 
        xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)



###########33
# KSD
###############
Random.seed!(1)
D_nuts = nuts(zeros(d), 0.7, logp, ∇logp, 5000, 20000)
ksd_nuts = ksd(D_nuts, ∇logp)
JLD.save(joinpath("result/","nuts.jld"), "sample", D_nuts, "ksd", "ksd_nuts")

n_lfrg = 30
μ=zeros(d)
D = 0.01*ones(d)
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

Random.seed!(1)
ksd_plot(o; μ = zeros(d), D = 0.01*ones(d), ϵ = 5e-4*ones(d), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample = 5000, title  = "Linear regression")