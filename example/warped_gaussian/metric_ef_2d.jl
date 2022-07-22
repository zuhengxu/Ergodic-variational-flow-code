using Flux, Zygote, JLD
include("model_2d.jl")
include("../../inference/ErgFlow/ergodic_flow.jl")
include("../../inference/HVI/HVI.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")

bw = 1
n_lfrg = 80
n_mcmc = 20
niters = 20000
ϵ0 = 0.0005*ones(2)
verb = 1000
ref, inv_ref= ErgFlow.pseudo_refresh_coord, ErgFlow.inv_refresh_coord
opt_args = (elbo_size = 20, niters = niters, learn_init = false, verbose_freq = verb, optimizer = Flux.ADAM(1e-3))
μ0, D0 = μ, D
D_nuts = nuts(μ, 0.7, logp, ∇logp, 1000, 10000)
ksd_nuts = ksd(D_nuts, ∇logp; bw = bw) #0.213862


# lap_mom = (ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std)
# # rough compare 
# n_lfrg = 50
# o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
#         ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
#         ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
# a = ErgFlow.HF_params(0.02*ones(2), μ, D) # using learned VI parameters
# D_ef, M, U = ErgFlow.Sampler(o,a,ErgFlow.pseudo_refresh_coord,500, 3000; nBurn = 10)
# ksd_ef = ksd(D_ef', ∇logp; bw = 0.05) #0.166959

# ErgFlow.ELBO(o, exp.(ps[1][1]), ps[1][2], ps[1][3], ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord,100; 
#                                 nBurn = 0, elbo_size = 1000, print = true)

# opt
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)
a = ErgFlow.HF_params(ϵ0, μ0, D0) # using std gaussian

ps, el, PS = ErgFlow.HamErgFlow(o,a,ref, inv_ref, n_mcmc; opt_args...)
# plot(el, ylim = (-20, 10))
ksd_ef = ErgFlow.KSD_trace(o, PS, ref, n_mcmc; learn_init = false, N = 1000, bw = bw)
JLD.save("result/metric_ef$(n_lfrg)_$(n_mcmc).jld", "ELBO", el, "KSD", ksd_ef, "ksd_nuts",ksd_nuts , "ps_trace", PS)

# o_his = HVI.HIS(d, n_lfrg, n_mcmc, logp, ∇logp, randn, logq, 
#         ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std)
# a_his = HVI.HIS_params(ϵ0, ones(n_mcmc), μ0, D0)
# ps_his, el_his, PS_his = HVI.his_vi(o_his, a_his;opt_args...)
# # plot(el_his, ylim = (-20, 10))
# ksd_his = HVI.KSD_trace(o_his, PS_his; μ = μ, D = D, N = 1000, bw = bw, learn_init = false)
# JLD.save("result/metric_his$(n_lfrg)_$(n_mcmc).jld", "ELBO", el_his, "KSD", ksd_his, "ps_trace", PS_his)

# o_uha = HVI.UHA(d, n_lfrg, n_mcmc, logp, ∇logp, randn, logq, ∇logq, 
#         ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std)
# a_uha = HVI.UHA_params(ϵ0, ones(n_mcmc-1), [0.], μ0, D0)
# ps_uha, el_uha, PS_uha = HVI.uha_vi(o_uha, a_uha; opt_args...)
# # plot(el_uha, ylim = (-20, 10))
# ksd_uha = HVI.KSD_trace(o_uha, PS_uha; μ = μ, D = D, N = 1000, bw = bw, learn_init = false)
# # HVI.Sampler(o_uha, a_uha, 100)
# JLD.save("result/metric_uha$(n_lfrg)_$(n_mcmc).jld", "ELBO", el_uha, "KSD", ksd_uha, "ps_trace", PS_uha)

# plot([ksd_ef, ksd_his, ksd_uha], lw = 3, alpha = 0.6 ,label = ["EF" "HIS" "UHA"])
# hline!([ksd_nuts], linestype=:dash, lw = 2)



ef = JLD.load("result/metric_ef$(n_lfrg)_$(n_mcmc).jld")
his = JLD.load("result/metric_his$(n_lfrg)_$(n_mcmc).jld")
uha = JLD.load("result/metric_uha$(n_lfrg)_$(n_mcmc).jld")
p = plot([ef["KSD"], his["KSD"], uha["KSD"]], lw = 3, alpha = 0.6 ,label = ["EF" "HIS" "UHA"])
hline!([ef["ksd_nuts"]], linestype=:dash, lw = 2)
savefig(p, joinpath("figure/","ksd$(n_lfrg)_$(n_mcmc).png"))
p1 = plot(ef["ELBO"], label = "EF")
p2 = plot(his["ELBO"], label = "HIS")
p3 = plot(uha["ELBO"], label = "UHA")
p_el =  plot(p1, p2, p3, layout = 3, title = "lfrg = $n_lfrg , ref = $n_mcmc")
savefig(p_el, joinpath("figure/","elbo$(n_lfrg)_$(n_mcmc).png"))
# X = 0.001:0.001:0.00d5
# Y = 0.001:0.001:0.005
# E1= ef_eps(X, Y, o; μ = μ, D = D, n_mcmc = 50)

# ef = JLD.load("result/metric_ef.jld")
# his = JLD.load("result/metric_his.jld")
# uha = JLD.load("result/metric_uha.jld")
# ef["ELBO"]
# p1 = plot(ef["ELBO"])
# p2 = plot(his["ELBO"])
# p3 = plot(uha["ELBO"])
# p_el =  plot(p1, p2, p3, layout = 3, title = "lfrg = $n_lfrg , ref = $n_mcmc")
# savefig(p_el, joinpath("figure/","elbo$(n_lfrg)_$(n_mcmc).png"))