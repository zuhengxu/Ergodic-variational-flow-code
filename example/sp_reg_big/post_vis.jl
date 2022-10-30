using Flux, Zygote, JLD, Plots, ProgressMeter
using ErgFlow
include("model.jl")
include("../../inference/SVI/svi.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/nf_train.jl")



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
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a_p = HF_params(8e-4*ones(d), μ, D) 
D_ef, _, _ = ErgFlow.Sampler(o, a_p, ErgFlow.pseudo_refresh_coord, 2000, 2000)
JLD.save("result/EF_sample.jld", "sample", D_ef)

###################
# logpost 4 conditionals density
###################
Random.seed!(1)
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

a_p = HF_params(8e-4*ones(d), μ, D) 

q0 = MvNormal(vcat(μ, zeros(d)), diagm(vcat(D.^2.0, ones(d))))
flow_nvp, _, _, _ = train_rnvp(q0, logp_nf, logq_nf, 2d, 200000)

xs = [0.:0.1:20 ;]
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