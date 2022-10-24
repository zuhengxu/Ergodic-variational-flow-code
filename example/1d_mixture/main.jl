using Distributions, ForwardDiff, ErgFlow

function logmix(x, w, logps)
    a = maximum(logps(x))
    wl = w.*exp.((logps(x) .- a))
    return a + log(sum(wl))
end
logps(x) = [logpdf(Normal(0, 0.8), x), logpdf(Normal(-3, 1.5), x), logpdf(Normal(3, 0.8), x)]
logp(x) = logmix(x, [0.3, 0.5, 0.2], logps)
pdf_mix(x) = expm1(logp(x)) + 1.0
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