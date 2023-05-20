ENV["JULIA_SCRATCH_TRACK_ACCESS"] = 0
println(Threads.nthreads())

using GPUCompiler
using CUDA
using Flux, Zygote, JLD, JLD2
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

n_mcmc = nothing
n_lfrg = nothing
pair = nothing

mcmcs = [5, 10]
lfrgs = [10, 20, 50]

grid = zeros(Int, size(mcmcs,1) * size(lfrgs,1), 2)

grid[:,1] = vec(repeat(mcmcs, 1, size(lfrgs,1))')
grid[:,2] = repeat(lfrgs, size(mcmcs,1))

if size(ARGS,1) > 0
    pair = parse(Int, ARGS[1])
    n_mcmc = grid[pair, 1]
    n_lfrg = grid[pair, 2]
else
    n_mcmc = 10
    n_lfrg = 50
end

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
El_uha, Ksd_uha = uha_eval(PS[1], (sample_q0, logq0, ∇logq0, ∇logp, n_mcmc, d, n_lfrg), ∇logp, 10000)

cd("/scratch/st-tdjc-1/mixflow")
if size(ARGS,1) > 0
    JLD.save("uha_heavy_reg_"*string(pair)*".jld", "PS", PS[1], "elbo", El_uha, "ksd", Ksd_uha, "time", t)
else
    JLD.save("uha_heavy_reg.jld", "PS", PS[1], "elbo", El_uha, "ksd", Ksd_uha, "time", t)
end

