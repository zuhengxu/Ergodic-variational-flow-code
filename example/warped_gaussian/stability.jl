include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")
import PlotlyJS as pjs
Random.seed!(1)
###########3
# ELBO
###########
n_lfrg = 80
o_lap = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

o_norm = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)

a = ErgFlow.HF_params(0.003*ones(d), μ, D)


stability_plot(o_lap, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]; nsample = 100, res_name = "stab_lap.jld")
stability_plot(o_norm, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000]; nsample = 100, res_name = "stab_norm.jld")

