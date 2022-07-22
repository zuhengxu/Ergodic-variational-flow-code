include("model_2d.jl")
include("../common/plotting.jl")
include("../common/result.jl")
include("../common/error.jl")
import PlotlyJS as pjs
Random.seed!(1)
###########3
#  fwd/bwd flow err
###########
n_lfrg = 50
o_lap = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

o_norm = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

o_log = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randlogistic, ErgFlow.lpdf_logistic, ErgFlow.∇lpdf_logistic, ErgFlow.cdf_logistic, ErgFlow.invcdf_logistic, ErgFlow.pdf_logistic,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

a = ErgFlow.HF_params(0.016*ones(d), μ, D)


stability_plot(o_lap, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_lap.jld")
stability_plot(o_norm, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_norm.jld")
stability_plot(o_log, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_log.jld")



