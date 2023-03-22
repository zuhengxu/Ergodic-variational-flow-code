using Flux, Zygote, JLD, JLD2, Plots
using ErgFlow
include("model.jl")
include("../../inference/util/ksd.jl")


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
###############
# train UHA
###############
Random.seed!(1)
n_mcmc = 10
n_lfrg = 20
elbo_size = 5
niters = 10000
ϵ0 = 0.001*ones(d)
logit_T0_uha = ones(n_mcmc-1)
logit_η0 = [0.5]


PS, el,_ = uha_vi(sample_q0::Function, logp::Function, logq::Function, ∇logq::Function, ∇logp::Function,
                n_mcmc::Int, n_lfrg::Int, niters::Int, d::Int, elbo_size::Int, 
                ϵ0::Vector{Float64}, logit_T0_uha::Vector{Float64}, logit_η0::Vector{Float64}; 
                optimizer = Flux.ADAM(1e-3))