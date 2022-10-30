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
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5], flow_type="RealNVP", nrun = 5, file_name = "RealNVP5_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [8], flow_type="RealNVP", nrun = 5, file_name = "RealNVP8_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [10], flow_type="RealNVP", nrun = 5, file_name = "RealNVP10_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Planar", nrun = 5, file_name = "Planar_run.jld")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Radial", nrun = 5, file_name = "Radial_run.jld")
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 5, flow_type="RealNVP", seed = 3)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 10, flow_type="Planar", seed = 3)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 5, flow_type="Radial", seed = 3)

###################
# compute ksd using NF samples
###################
files = ["RealNVP5.jld","Planar10.jld", "Radial5.jld"]
nf_ksd(files, o)
