using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")


############
# NUTS
############
Random.seed!(1)
D_nuts = nuts(μ, 0.7, logp, ∇logp, 5000, 20000)
ksd_nuts = ksd(D_nuts, ∇logp)
JLD.save(joinpath("result/","nuts.jld"), "sample", D_nuts, "ksd", ksd_nuts)

###############
# ErgFLow
################
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
n_lfrg = 30
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a = HF_params(1.1e-3*ones(d), μ, D) 
D_ef, _, _ = ErgFlow.Sampler(o, a, ErgFlow.pseudo_refresh_coord, 600, 5000)
JLD.save("result/EF_sample.jld", "sample", D_ef)
