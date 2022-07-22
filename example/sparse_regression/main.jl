using Flux, Zygote, JLD, Plots
include("model.jl")
include("../../inference/ErgFlow/ergodic_flow.jl")
include("../../inference/HVI/HVI.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")

### fit MF Gaussian
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 20000; elbo_size = 10, logging_ps = false)
# Plots.plot(el1, ylims = (-50, 10))
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, zeros(d), ones(d); elbo_size = 10000)

###########
## EF
###########
n_lfrg = 30
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
Random.seed!(1)
ELBO_plot(o, o1; μ=zeros(d), D = 0.01*ones(d), eps = [0.0001, 0.0005, 0.001], Ns = [100, 500, 1000,1500], nBs = [0,0,0,0,0,0], elbo_size = 1000, 
        res_name = "el2.jld",fig_name = "lr_elbo2.png", title = "Linear regression", 
        xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)


# ELBO= JLD.load("result/el.jld")
# ELBO["elbos"]
# plot([0, 100, 200, 500, 1000], ELBO["elbos"]')


Random.seed!(1)
D_nuts = nuts(0.1*ones(d), 0.7, logp, ∇logp, 5000, 10000)
ksd_nuts = ksd(D_nuts, ∇logp)
JLD.save(joinpath("result/","nuts.jld"), "sample", D_nuts, "ksd", "ksd_nuts")
# 3.07

n_lfrg = 30
μ=zeros(d)
D = 0.01*ones(d)
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a = ErgFlow.HF_params(5e-4*ones(d), μ, D) 
# n_mcmc = 1500
# nsample = 5000 
# Random.seed!(1)
# T = ErgFlow.Sampler(o, a, ErgFlow.pseudo_refresh_coord, n_mcmc, nsample; nBurn = 0)[1]
# ksd_ef= ksd(T, o.∇logp)
# println("ksd = $ksd_ef")
# # 2e-3 -> 3.36056
# # 1.2e-3 -> 3.80
# #5e-4 -> 7.0
# # 3e-4 -> 33

# T_init =  1e-2*ones(d)' .* randn(5000, d) 
# ksd_norm = ksd(T_init, o.∇logp)
#511.606


# Random.seed!(1)
# ksd_plot(o; μ = zeros(d), D = 0.01*ones(d), ϵ = 5e-4*ones(d), Ns = [100, 200, 500, 1000, 1500], nBs = [0], nsample = 5000, title  = "Sparse regression")
