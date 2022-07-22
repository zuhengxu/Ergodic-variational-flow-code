include("model_multi_d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")

#########
# d = 5
#########
# d = 5
# σ² = 36
# Z = sqrt(σ² * (2*π)^d) # normalizing constant
# # x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) 
# logp(x) = -log(Z) - 0.5 * (x[1]-σ²/4)^2/σ² - 0.5 * (x[2:end]' * x[2:end]) / exp(0.5*x[1]) - (d-1)*x[1]/4
# ∇logp(x) = vcat(-(x[1]-σ²/4)/σ² + 0.25 * x[2:end]' * x[2:end] * exp(-x[1]/2) - (d-1)/4, -x[2:end] .* exp(-x[1]/2))

# # fit MF Gaussian
# o1 = SVI.MFGauss(d, logp, randn, logq)
# a1 = SVI.mf_params(zeros(d), ones(d)) 
# ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
# μ,D = ps1[1][1], ps1[1][2]
# el_svi = SVI.ELBO(o1, μ, D; elbo_size = 10000)

# Random.seed!(1)
# n_lfrg = 80
# o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
#         ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
#         ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 

# Random.seed!(1)
# ELBO_plot(o, o1;μ=μ, D=D, eps = [0.001, 0.009, 0.018], Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0, 0, 0, 0, 0, 0], elbo_size = 1000, 
#                fig_name = "5d_elbo.png", res_name = "5d_elbo_dat.jld")

# Random.seed!(1)
# els = eps_tunning([0.001:0.002:0.025 ;],o; μ = μ, D = D, n_mcmc = 1000, elbo_size=1000,fig_name = "5d_funnel_tune.png", res_name = "5d_eps.jld", title = "Neal's Funnel", 
#                 xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18,xrotation = 20)
# # 0.009 is optimal

# Random.seed!(1)
# ksd_plot(o; μ = μ, D = D, ϵ = 0.009*ones(d), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  =2000, title  = "Neal's Funnel", fig_name = "5d_funnel_ksd.png", res_name = "5d_ksd.jld")

# Random.seed!(1)
# D_nuts = nuts(μ, 0.7, logp, ∇logp, 5000, 10000)
# ksd_nuts = ksd(D_nuts, ∇logp) # 0.02120979813481245, bw ≈ 3678

#########
# d = 20
#########
d = 20
σ² = 36
Z = sqrt(σ² * (2*π)^d) # normalizing constant
# x1∼N(σ²/4, σ²), x2∼N(0, exp(x1/2)) 
logp(x) = -log(Z) - 0.5 * (x[1]-σ²/4)^2/σ² - 0.5 * (x[2:end]' * x[2:end]) / exp(0.5*x[1]) - (d-1)*x[1]/4
∇logp(x) = vcat(-(x[1]-σ²/4)/σ² + 0.25 * x[2:end]' * x[2:end] * exp(-x[1]/2) - (d-1)/4, -x[2:end] .* exp(-x[1]/2))

# fit MF Gaussian
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 50000; elbo_size = 1, logging_ps = false)
μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ, D; elbo_size = 10000)

Random.seed!(1)
n_lfrg = 100
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 

Random.seed!(1)
ELBO_plot(o, o1;μ=μ, D=D, eps = [0.0005, 0.001, 0.01], Ns = [100, 500, 1000, 1500, 2000], nBs = [0, 0, 0, 0, 0, 0], elbo_size = 2000, 
               fig_name = "20d_elbo.png", res_name = "20d_elbo_dat.jld")

# Random.seed!(1)
# ErgFlow.ELBO(o, 0.001*ones(d), μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord, 1000; 
#                         nBurn = 0, elbo_size = 1000, print = true)

# 0.001 -> -2.3328749674798845
# 0.005 -> -1.5617397138377789
# 0.009 -> -2.090939179119401
# 0.015 -> -4.296334973150374

# Random.seed!(1)
# els = eps_tunning(vcat([0.0005], [0.001:0.001:0.005 ;]), o; μ = μ, D = D, n_mcmc = 2000, elbo_size=5000,fig_name = "20d_funnel_tune.png", res_name = "20d_eps.jld", title = "Neal's Funnel", 
#                 xtickfont=font(18), ytickfont=font(18), guidefont=font(18), legendfont=font(18), titlefontsize = 18,xrotation = 20)

# Random.seed!(1)
# ksd_plot(o; μ = μ, D = D, ϵ = 0.001*ones(d), Ns = [100, 200, 500, 1000, 1500, 2000], nBs = [0], nsample  = 5000, title  = "Neal's Funnel", fig_name = "20d_funnel_ksd.png", res_name = "20d_ksd.jld")

# 0.0005 -> 0.0971644
# 0.001 -> 0.0600964
# 0.002 -> 0.0605568
# 0.005 -> 0.11

# T_init =  μ'.+ D' .* randn(2000, d) 
# ksd(T_init, ∇logp) 

# Random.seed!(1)
# D_nuts = nuts(μ, 0.7, logp, ∇logp, 5000, 20000)
# ksd_nuts = ksd(D_nuts, ∇logp) # 0.15415839361629038