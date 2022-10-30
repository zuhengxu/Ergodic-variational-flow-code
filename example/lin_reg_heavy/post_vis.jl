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
# el_svi = SVI.ELBO(o1, μ,D; elbo_size = 10000)
# JLD.save(joinpath("result/", "mfvi.jld"), "μ", μ, "D", D, "elbo", el_svi)

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

############
# NUTS
############
Random.seed!(1)
D_nuts = nuts(μ, 0.7, logp, ∇logp, 10000, 20000)
# ksd_nuts = ksd(D_nuts, ∇logp)
JLD.save(joinpath("result/","nuts_sample.jld"), "sample", D_nuts)

##################
# ergflow
####################
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a_p = HF_params(3e-5*ones(d), μ, D) 
D_ef, _, _ = ErgFlow.Sampler(o, a_p, ErgFlow.pseudo_refresh_coord, 2000, 2000)
JLD.save("result/EF_sample.jld", "sample", D_ef)

###################
# pairwise plot
###################
Onuts = JLD.load("result/nuts_sample.jld")
NF_planar = JLD.load("result/planar.jld")

D_nuts = Onuts["sample"]
D_nf = NF_planar["scatter"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]


####################
# visualization
####################
# in this example, look at conditional post logpdf and lpdf estimated by NF/ergflow


# post pairwise kde +  scatter from NF and EF
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
    p_vis = pairplots(D_nuts[:, idx], D_ef[1:1000, idx], D_nf[1:1000, idx])
    savefig(p_vis, "figure/post_vis"*"$(i).png")
end

####################
# visualization
####################
# in this example, look at conditional post logpdf and lpdf estimated by NF/ergflow

# train NF (pick the best one)
Random.seed!(1)
q0 = MvNormal(vcat(μ, zeros(d)), diagm(vcat(D.^2.0, ones(d))))

# Planar
F = ∘([PlanarLayer(2d) for i in 1:5]...)
flow = transformed(q0, F)
_, el, ps = nf(flow, logp_nf, logq_nf, 100000; elbo_size = 10)

# Radial
# F = ∘([RadialLayer(2d) for i in 1:5]...)
# flow = transformed(q0, F)
# _, el, ps = nf(flow, logp_nf, logq_nf, 100000; elbo_size = 10)

# RealNVP
# flow, _, _, _ = train_rnvp(q0, logp_nf, logq_nf, 2d, 100000; hdims = 20, elbo_size = 10)

# log pdf of last dimension
xs = [-10:0.1:10 ;]
lpdf_nf = zeros(size(xs, 1))
lpdf_joint = zeros(size(xs, 1))
lpdf_ef = zeros(size(xs, 1))

for i in 1:size(xs, 1)
    zss = zeros(2*d)
    zss[d] = xs[i]
    lpdf_nf[i] = logpdf(flow, zss)

    zss = zeros(d)
    zss[d] = xs[i]
    lpdf_joint[i] = o.logp(zss)
end

ϵ = 2.5e-5*ones(d)
n_mcmc = 2500
nBurn = 0
inv_ref = ErgFlow.inv_refresh_coord
prog_bar = ProgressMeter.Progress(size(xs, 1), dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
@threads for i in 1:size(xs, 1)
    zss = zeros(d)
    zss[d] = xs[i]
    lpdf_ef[i], _ = ErgFlow.log_density_est(zss, zeros(d), rand(), o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn)
    ProgressMeter.next!(prog_bar)
end

plot1 = plot(xs, lpdf_nf, label="NF")
plot!(xs, lpdf_ef, label="EF")
plot2 = plot(xs, lpdf_joint, label="logp")
p = plot(plot1, plot2, layout = (1, 2), legend = true)

savefig(p, "figure/univariate_lpdf.png")

# function p_lpdf(omg = 1)
#     xs = [-30:0.1:30 ;]
#     lpdf_joint = zeros(size(xs, 1))

#     for i in 1:size(xs, 1)
#         zss = zeros(d)
#         zss[omg] = xs[i]
#         lpdf_joint[i] = o.logp(zss)
#     end

#     plot(xs, lpdf_joint)
# end