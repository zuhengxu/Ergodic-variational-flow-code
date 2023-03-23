using Flux, Zygote, JLD, JLD2, Plots
using ErgFlow
include("model.jl")
include("../../inference/util/ksd.jl")
include("../../inference/mcvae/hvi.jl")


#######
# setting
#######
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
logq0(z) = logq(z, μ, D)
∇logq0(z) = ∇logq(z, μ, D)

function sample_q0(n)
    if n == 1
        return D .* randn(d) .+ μ 
    else
        return D .* randn(n, d) .+ μ
    end
end
sample_q0() = sample_q0(1)

###############
# train UHA
###############
Random.seed!(1)
n_mcmc = 10
n_lfrg = 20
elbo_size = 5
niters = 20000
ϵ0 = 0.001*ones(d)
logit_T0_uha = ones(n_mcmc-1)
logit_η0 = [0.5]

tick();
PS, el,_ = uha_vi(sample_q0::Function, logp::Function, logq0::Function, ∇logq0::Function, ∇logp::Function,
                n_mcmc::Int, n_lfrg::Int, niters::Int, d::Int, elbo_size::Int, 
                ϵ0::Vector{Float64}, logit_T0_uha::Vector{Float64}, logit_η0::Vector{Float64}; 
                optimizer = Flux.ADAM(1e-3))
t = tok()
#####################
# eval uha
#####################
Random.seed!(1)
El_uha, Ksd_uha = uha_eval(PS[1], (sample_q0, logq0, ∇logq0, ∇logp, n_mcmc, d, n_lfrg), ∇logp, 5000)
JLD.save("result/uha.jld", "PS", PS[1], "elbo", El_uha, "ksd", Ksd_uha, "time", t)

