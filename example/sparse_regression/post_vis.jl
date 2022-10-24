using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")



######################
# ### fit MF Gaussian
######################
Random.seed!(1)
o1 = SVI.MFGauss(d, logp, randn, logq)
a1 = SVI.mf_params(zeros(d), ones(d)) 
ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false, optimizer = Flux.ADAM(1e-3))
# Plots.plot(el1, ylims = (-50, 10))

μ,D = ps1[1][1], ps1[1][2]
el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000)
JLD.save(joinpath("result/", "mfvi.jld"), "μ", μ, "D", D, "elbo", el_svi)

###########
##  NF
###########
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
n_lfrg = 30
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a = HF_params(1.1e-3*ones(d), μ, D) 

# joint target and joint init
logp_nf(x) = o.logp(x[1:d]) + o.lpdf_mom(x[d+1:end])
μ_joint = vcat(μ, zeros(d))
D_joint = vcat(D, ones(d))
logq_nf(x) =  -0.5*2d*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))

@info "running single nf"
single_nf(logp_nf, logq_nf, μ, D, d; niter = 100000, flow_type = "Planar", file_name = "planar.jld")
el_nf = JLD.load("result/planar.jld")["elbo"]
println(el_nf)

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
D_ef, _, _ = ErgFlow.Sampler(o, a, ErgFlow.pseudo_refresh_coord, 600, 5000)
JLD.save("result/EF_sample.jld", "sample", D_ef)

###################
# pairwise plot
###################
NUTS = JLD.load("result/nuts.jld")
NF_planar = JLD.load("result/planar.jld")

D_nuts = NUTS["sample"]
D_nf = NF_planar["scatter"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]

p_vis = pairplots(D_nuts, D_ef[1:1000, :], D_nf[1:1000, :])
savefig(p_vis, "figure/post_vis.png")