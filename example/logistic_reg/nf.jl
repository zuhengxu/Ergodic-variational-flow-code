using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/SVI/svi.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")



###########
##  NF setup
###########
# Random.seed!(1)
# o1 = SVI.MFGauss(d, logp, randn, logq)
# a1 = SVI.mf_params(zeros(d), ones(d)) 
# ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false, optimizer = Flux.ADAM(1e-3))
# # Plots.plot(el1, ylims = (-50, 10))

# μ,D = ps1[1][1], ps1[1][2]
# el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000)
# JLD.save(joinpath("result/", "mfvi.jld"), "μ", μ, "D", D, "elbo", el_svi)
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

# joint target and joint init
logp_nf(x) = o.logp(x[1:d]) + o.lpdf_mom(x[d+1:end])
μ_joint = vcat(μ, zeros(d))
D_joint = vcat(D, ones(d))
logq_nf(x) =  -0.5*2d*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))


# run 5 runs for each layer
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [10], flow_type="RealNVP", nrun = 5, file_name = "RealNVP8_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [8], flow_type="RealNVP", nrun = 5, file_name = "RealNVP8_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5], flow_type="RealNVP", nrun = 5, file_name = "RealNVP5_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Planar", nrun = 5, file_name = "Planar_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Radial", nrun = 5, file_name = "Radial_run.jld")
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 8, flow_type="RealNVP", seed = 3)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 5, flow_type="Planar", seed = 2)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 20, flow_type="Radial", seed = 2)
###################
# compute ksd using NF samples
###################
files = ["RealNVP8.jld","Planar5.jld", "Radial20.jld"]
nf_ksd(files, o)