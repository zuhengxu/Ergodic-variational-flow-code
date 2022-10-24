using Distributions, ForwardDiff, ErgFlow

# 1d-gaussian target
logp(x) = logpdf(Normal(2, 2), x)
∇logp(x) = ForwardDiff.derivative(logp, x) 
# init struct 
Flow = ErgFlow.HamFlow_1d(50, ErgFlow.lpdf_normal, randn, ∇logp, 
                    ErgFlow.randl, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, 
                    ErgFlow.stream_x, ErgFlow.mixer, ErgFlow.inv_mixer)
λ = ErgFlow.HF1d_params(0.05)

# hyperparams
z0, ρ0, u0 = 0.5, 1.0, 0.5
n_mcmc = 100
N = 10000


folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 