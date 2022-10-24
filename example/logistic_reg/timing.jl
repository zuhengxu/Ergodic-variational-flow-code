using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/SVI/svi.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")
include("../common/efficiency.jl")


MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

n_lfrg = 50     
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a_p = HF_params(2e-3*ones(d), μ, D) 

n_mcmc = 500
run_time_per_sample(o, a_p, ErgFlow.pseudo_refresh_coord; n_run = 100, n_mcmc = n_mcmc, seed = 2022)
ess_time(o, a_p, pseudo_refresh_coord; num_trials = 10, nsamples = 2000, n_mcmc = n_mcmc)