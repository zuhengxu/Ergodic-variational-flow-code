using LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions, Parameters
using ProgressMeter, Flux
include("../util/train.jl")

struct MFGauss <: StochasticVI
    d::Int64
    # target 
    logp::Function # log target density
    # VI distribution
    q_sampler::Function
    logq::Function
end

struct mf_params<: params
    μ::Vector{Float64}
    D::Vector{Float64}
end

function single_elbo(o::MFGauss, μ, D)
    z = D.*o.q_sampler(o.d) .+ μ
    el = o.logp(z) - o.logq(z, μ, D) 
    return el
end

function ELBO(o::MFGauss, μ, D; elbo_size = 1)
    el = 0.0    
    @simd for i in 1:elbo_size
        el += 1/elbo_size*single_elbo(o, μ, D)
    end
    return el
end

function vi(o::MFGauss, a::mf_params, niters::Int; elbo_size::Int =1, optimizer = Flux.ADAM(1e-3), kwargs...)
    μ, D = a.μ, a.D
    ps = Flux.params(μ, D)

    #define loss
    loss = () -> begin 
        elbo = ELBO(o, μ, D; elbo_size = elbo_size)
        return -elbo
    end

    elbo_log, ps_log = vi_train!(niters, loss, ps, optimizer; kwargs...)
    return [[copy(p) for p in ps]], -elbo_log, ps_log

end
