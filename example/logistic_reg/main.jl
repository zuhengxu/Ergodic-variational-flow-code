using Flux, Zygote, JLD, Plots
using ErgFlow
include("model.jl")
# include("../../inference/HVI/HVI.jl")
include("../../inference/MCMC/NUTS.jl")
# include("../../inference/util/metric.jl")
include("../../inference/util/ksd.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")

# ### fit MF Gaussian
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false, optimizer = Flux.ADAM(1e-3))

μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000)



###########
## EF
###########
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)


# Random.seed!(1)
# ELBO_plot(o, o1; μ= μ, D = D, eps = [1e-4, 2e-3, 4e-3], Ns = [100, 200, 500, 1000, 1500], nBs = [0,0,0,0,0,0], elbo_size = 1000, 
#         res_name = "el.jld",fig_name = "lg_elbo.png", title = "Logistic regression", 
#         xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18, xrotation = 20)

# # Random.seed!(1)
# # D_nuts = nuts(μ, 0.7, logp, ∇logp, 5000, 20000)
# # ksd_nuts = ksd(D_nuts, ∇logp)
# # 0.34


# Random.seed!(1)
# ksd_plot(o; μ = μ, D = D, ϵ = 2e-3*ones(d), Ns = [100, 200, 500, 1000, 1500], nBs = [0], nsample = 5000, title  = "Logistic regression")

###########
##  NF
###########
# joint target and joint init
logp_nf(x) = o.logp(x[1:d]) + o.lpdf_mom(x[d+1:end])
μ_joint = vcat(μ, zeros(d))
D_joint = vcat(D, ones(d))
logq_nf(x) =  -0.5*2d*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))
# logq_joint(x) =  -0.5*4*log(2π) - 0.5*sum(abs2, x)

@info "running single nf"
single_nf(logp_nf, logq_nf, μ, D, d; niter = 100000)
el_nf = JLD.load("result/nf.jld")["elbo"]
println(el_nf)
