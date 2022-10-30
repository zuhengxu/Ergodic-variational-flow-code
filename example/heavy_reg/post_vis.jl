using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/SVI/svi.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")


######################
# ### fit MF Gaussian
######################
# Random.seed!(1)
# o1 = SVI.MFGauss(d, logp, randn, logq)
# a1 = SVI.mf_params(zeros(d), ones(d)) 
# ps1, el1,_ = SVI.vi(o1, a1, 100000; elbo_size = 1, logging_ps = false, optimizer = Flux.ADAM(1e-3))
# # Plots.plot(el1, ylims = (-50, 10))

# μ,D = ps1[1][1], ps1[1][2]
# el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000) # -409
# JLD.save(joinpath("result/", "mfvi.jld"), "μ", μ, "D", D, "elbo", el_svi)

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

############
# NUTS
############
Random.seed!(1)
D_nuts = nuts(μ, 0.7, logp, ∇logp, 10000, 20000)
ksd_nuts = ksd(D_nuts[1:5000, :], ∇logp)
JLD.save(joinpath("result/","nuts_big.jld"), "sample", D_nuts, "ksd", ksd_nuts)

##################
# ergflow
####################
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a_p = HF_params(3e-4*ones(d), μ, D) 
D_ef, _, _ = ErgFlow.Sampler(o, a_p, ErgFlow.pseudo_refresh_coord, 1000, 1000)
JLD.save("result/EF_sample.jld", "sample", D_ef)

###################
# pairwise plot
###################
NUTS = JLD.load("result/nuts_big.jld")
NF_nvp = JLD.load("result/RealNVP5.jld")

D_nuts = NUTS["sample"]
D_nf = NF_nvp["Samples"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]

# psot pairwise kde
for i in 1:Int(ceil(d/10))
    k = i<Int(ceil(d/10)) ? 10*i : d
    idx = [10*(i-1)+1:k ;]
    p_kde = pairkde(D_nuts[:, idx])
    savefig(p_kde, "figure/post_kde"*"$(i).png")
end

# post pairwise kde +  scatter from NF and EF
for i in 1:Int(ceil(d/10))
    k = i<Int(ceil(d/10)) ? 10*i : d
    idx = [10*(i-1)+1:k ;]
    p_vis = pairplots(D_nuts[:, idx], D_ef[1:2000, idx], D_nf[1:2000, idx])
    savefig(p_vis, "figure/post_vis"*"$(i).png")
end

###################
# logpost 4 conditionals density
###################
Random.seed!(1)
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
logp_nf(x) = o.logp(x[1:d]) + o.lpdf_mom(x[d+1:end])
μ_joint = vcat(μ, zeros(d))
D_joint = vcat(D, ones(d))
logq_nf(x) =  -0.5*2d*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a_p = HF_params(8e-4*ones(d), μ, D) 

q0 = MvNormal(vcat(μ, zeros(d)), diagm(vcat(D.^2.0, ones(d))))
flow_nvp, _, _, _ = train_rnvp(q0, logp_nf, logq_nf, 2d, 200000)

xs = [-10:0.1:10 ;]
lpdf_nf = zeros(size(xs, 1))
lpdf_joint = zeros(size(xs, 1))
lpdf_ef = zeros(size(xs, 1))

coord = 1
for i in 1:size(xs, 1)
    zss = zeros(2*d)
    zss[coord] = xs[i]
    lpdf_nf[i] = logpdf(flow_nvp, zss)

    zss = zeros(d)
    zss[coord] = xs[i]
    lpdf_joint[i] = o.logp(zss)
end

n_mcmc = 1000
inv_ref = ErgFlow.inv_refresh
prog_bar = ProgressMeter.Progress(size(xs, 1), dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
@threads for i in 1:size(xs, 1)
    zss = zeros(d)
    zss[coord] = xs[i]
    lpdf_ef[i], _ = ErgFlow.log_density_est(zss, zeros(d), 0.5, o, a_p.leapfrog_stepsize, a_p.μ, a_p.D, inv_ref, n_mcmc; nBurn = 0)
    ProgressMeter.next!(prog_bar)
end

JLD.save(joinpath("result/","univariate_lpdf.jld"), "xs", xs, "lpdf_nf", lpdf_nf, "lpdf_joint", lpdf_joint, "lpdf_ef", lpdf_ef)