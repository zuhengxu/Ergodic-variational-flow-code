using Flux, Zygote, JLD, JLD2, Plots, ProgressMeter
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
a_p = HF_params(8e-4*ones(d), μ, D) 

# joint target and joint init
logp_nf(x) = o.logp(x[1:d]) + o.lpdf_mom(x[d+1:end])
μ_joint = vcat(μ, zeros(d))
D_joint = vcat(D, ones(d))
logq_nf(x) =  -0.5*2d*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))




# run 5 runs for each layer
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5], hdims = 20, flow_type="RealNVP", nrun = 5, file_name = "RealNVP5_run.jld2")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [8], hdims = 20, flow_type="RealNVP", nrun = 5, file_name = "RealNVP8_run.jld2")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Planar", nrun = 5, file_name = "Planar_run.jld2")
tune_nf(logp_nf, logq_nf, μ, D, d; nlayers = [5, 10, 20], flow_type="Radial", nrun = 5, file_name = "Radial_run.jld2")
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 5, hdims = 20, flow_type="RealNVP", seed = 1)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 5, flow_type="Planar", seed = 1)
single_nf(logp_nf, logq_nf, μ, D, d; nlayers = 20, flow_type="Radial", seed = 1)
###################
# compute ksd using NF samples
###################
files = ["RealNVP5.jld2","Planar5.jld2", "Radial20.jld2"]
nf_ksd(files, o)