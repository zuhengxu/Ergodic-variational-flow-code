using LinearAlgebra, Distributions, Random, Plots, StatsBase, SpecialFunctions, Parameters
using ProgressMeter, Flux, Bijectors, TickTock
include("../util/train.jl")


function single_elbo(flow::Bijectors.MultivariateTransformed, logp, logq)
    x, y, logjac, logpdf_y = Bijectors.forward(flow)
    el = logp(y) -logq(x) + logjac
    # el = logp(y) - logpdf_y
    return el
end

function nf_ELBO(flow::Bijectors.MultivariateTransformed, logp, logq; elbo_size = 1)
    el = 0.0    
    @simd for i in 1:elbo_size
        el += 1/elbo_size*single_elbo(flow, logp, logq)
    end
    return el
end

function nf(flow::Bijectors.MultivariateTransformed, logp, logq, niters::Int; elbo_size::Int = 1, optimizer = Flux.ADAM(1e-3), kwargs...)

    ps = Flux.params(flow)

    #define loss
    loss = () -> begin 
        elbo = nf_ELBO(flow, logp, logq; elbo_size = elbo_size)
        return -elbo
    end

    elbo_log, ps_log = vi_train!(niters, loss, ps, optimizer; kwargs...)
    # time = peektimer()
    
    return [[copy(p) for p in ps]], -elbo_log, ps_log
end

include("realnvp.jl")

