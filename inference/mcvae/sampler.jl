# include("elbo.jl")
# include("../../util/result.jl")
using ProgressMeter
include("../util/ksd.jl")

function his_sample_single(sample_q0::Function, ∇logp::Function, T_alpha::Vector{Float64}, ϵ::Vector{Float64}, 
                            K::Int, d::Int, n_lfrg::Int)

    ρ = randn(d) #sample from standard normal
    z = sample_q0()
    β0_sqrt = prod(T_alpha) + 1e-8
    ρ ./= β0_sqrt 
    
    for id = 1:K
        # no refresh but tempering momentum
        his_one_transition!(∇logp, T_alpha, id, ρ, z, ϵ, n_lfrg)
    end

    return z, ρ 
end


function his_one_transition!(∇logp::Function, T_alpha::Vector{Float64}, id::Int, ρ_current, z_current, 
                                ϵ::Vector{Float64}, n_lfrg::Int)

    for i in 1:n_lfrg 
        ρ_current .+= 0.5 .* ϵ .* ∇logp(z_current) 
        z_current .+= ϵ .* ρ_current
        ρ_current .+= 0.5 .* ϵ .* ∇logp(z_current) 
    end 

    ρ_current .*= T_alpha[id]
end


function his_sample_with_elbo(logq::Function, sample_q0::Function, ∇logp::Function, logp_elbo::Function, n_subsample_elbo::Int,
                            T_alpha::Vector{Float64}, ϵ::Vector{Float64}, 
                            K::Int, d::Int, n_lfrg::Int; n_samples = 100)

    ρ = randn(n_samples, d) #sample from standard normal
    z = sample_q0(n_samples)
    E = Vector{Float64}(undef, n_samples)
    β0_sqrt = prod(T_alpha)  +1e-8
    ρ ./= β0_sqrt 
    # progress bar
    prog_bar = ProgressMeter.Progress(n_samples, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    
    for i = 1:n_samples
        el = -logq(@view(z[i,:])) + 0.5* β0_sqrt^2.0*dot(@view(ρ[i,:]), @view(ρ[i,:]))
        for id = 1:K
            # no refresh but tempering momentum
            his_one_transition!(∇logp, T_alpha, id, @view(ρ[i,:]), @view(z[i,:]), ϵ, n_lfrg)
        end
        # compute elbo
        E[i] = logp_elbo(@view(z[i,:]), n_subsample_elbo) - 0.5*dot(@view(ρ[i,:]), @view(ρ[i,:])) + el 
        ProgressMeter.next!(prog_bar)
    end
    return mean(E), z 
end

function uha_sample_single(sample_q0::Function, ∇logγ::Function, Temp_sched::Vector{Float64}, ϵ::Vector{Float64}, η::Vector{Float64}, 
                            K::Int, d::Int, n_lfrg::Int)
    
    z, ρ = sample_q0(), randn(d)

    for id = 1:K 
        uha_one_transition_for_sample!(∇logγ, Temp_sched, id, z, ρ, ϵ, η, n_lfrg)
    end

    return z, ρ 
end
     

function uha_one_transition_for_sample!(∇logγ::Function, T::Vector{Float64}, id::Int, z_current, ρ_current, 
                                        ϵ::Vector{Float64}, η::Vector{Float64}, n_lfrg::Int)
    
    logW_update = 0.5*(ρ_current' *ρ_current) 
    # leapfrog transition  
    for i in 1:n_lfrg
        ρ_current .+= 0.5 .* ϵ .* ∇logγ(z_current, T[id]) 
        z_current .+= ϵ .* ρ_current
        ρ_current .+= 0.5 .* ϵ .* ∇logγ(z_current, T[id]) 
    end
    
    # partially resample momentum
    u = randn(size(ρ_current, 1))
    @. ρ_current = sqrt(1.0 - η^2.0)*u + η*ρ_current
    
    # update logw
    logW_update -= 0.5*(ρ_current' *ρ_current) 
    return logW_update
end 


function uha_sample_with_elbo(logq::Function, sample_q0::Function, ∇logp::Function, ∇logq::Function, logp::Function, 
                            Temp_sched::Vector{Float64}, ϵ::Vector{Float64}, η::Vector{Float64}, 
                            K::Int, d::Int, n_lfrg::Int; n_samples = 100)

    z, ρ = sample_q0(n_samples), randn(n_samples,d)
    E = Vector{Float64}(undef, n_samples)
    ∇logγ(z, β) = β * ∇logp(z) .+ (1.0 - β) * ∇logq(z)
    
    # progress bar
    prog_bar = ProgressMeter.Progress(n_samples, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    for i = 1:n_samples
        logW = -logq(@view(z[i, :]))
        for id = 1:K 
            logw_upd = uha_one_transition_for_sample!(∇logγ, Temp_sched, id, @view(z[i, :]), @view(ρ[i,:]), ϵ, η, n_lfrg)
            logW += logw_upd
        end
        E[i] = logW + logp(@view(z[i,:]))
        ProgressMeter.next!(prog_bar)
    end

    return mean(E), z 
end


function reparam_uha(ps,logq, sample_q0, ∇logp, ∇logq, logp, n_mcmc, d, K)
    ϵ = @. expm1(ps[1]) + 1.0
    η = logistic.(ps[2])
    T = T_all(ps[3])
    return logq, sample_q0, ∇logp, ∇logq, logp, T, ϵ, η, n_mcmc, d, K
end

function HVI_progress(ps, HVIsampler::Function, argparse::Function, argparse_args,
                        kl_est::Function, kl_args, 
                        rel_err::Function, rel_args,
                        n_samples, ksd_est::Function, grd::Function)
    len = size(ps, 1)
    el = Vector{Float64}(undef, len)
    KLs = Vector{Float64}(undef, len)
    err_ms = Vector{Float64}(undef, len)
    err_covs = Vector{Float64}(undef, len)
    err_logs = Vector{Float64}(undef, len)
    EDs = Vector{Float64}(undef, len)
    KSDs = Vector{Float64}(undef, len)

    # progress bar
    prog_bar = ProgressMeter.Progress(len, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    for i = 1:len
        println("sample $i / $len")
        Args = argparse(ps[i], argparse_args...)
        # taking samples for fixed setting
        el[i], D = HVIsampler(Args...; n_samples)
        KLs[i] = kl_est(D, kl_args...)
        err_ms[i], err_covs[i], err_logs[i] = rel_err(rel_args..., D)
        EDs[i] = energy_dist(rel_args..., D)
        KSDs[i] = ksd_est(D, grd)
        
        # update progress bar
        ProgressMeter.next!(prog_bar)
    end
    return el, KLs, err_ms, err_covs, err_logs, EDs, KSDs
end



#############################################
# Functions for result processing (for mcvae)
#############################################
function argparse_uha(ps, sample_q0, logq, ∇logq, ∇logp, K, d, lf_n)
    ϵ = expm1.(ps[1]) .+ 1.0
    η = logistic.(ps[2])
    T = T_all(ps[3])
    return (sample_q0, logp, logq, ∇logq, ∇logp, T, ϵ, η, K, d, lf_n) 
end
#######################
#### generate repeated samples in a fixed setting (used for processing results)
#######################
function MCMCsampler(mcmc::Function, args; n_samples = 100)
    #=
    mcmc::Function: {ula_elbo, mala_elbo, his_elbo, uha_elbo}
    args...: arguments for mcmc function
    d: dim of sample
    n_sample: number of samples needed
    =#
    D = Vector{Vector{Float64}}(undef, n_samples)
    el = Vector{Float64}(undef, n_samples)
    prog_bar = ProgressMeter.Progress(n_samples, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    # use multithreading
    for i = 1:n_samples
        el[i],D[i] = mcmc(args...)
        ProgressMeter.next!(prog_bar)
    end

    # return final samples
    M = reduce(hcat,D)
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    Dat = size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
    return mean(el), Dat
end

function uha_eval(ps, argparse_args, ∇logp, n_samples)
    Args = argparse_uha(ps, argparse_args...)
    # taking samples for fixed setting
    el, D = MCMCsampler(uha_elbo, Args; n_samples = n_samples)
    Ksd = ksd(D, ∇logp)
    return el, Ksd
end

# computing both KSD and KL for given setting
function MCMC_progress(ps, mcmc::Function, argparse::Function, argparse_args,
                        ksd_est::Function, grd::Function, kl_est::Function, kl_args, 
                        rel_err::Function, rel_args,
                        n_samples)

    len = size(ps, 1)
    el = Vector{Float64}(undef, len)
    KSDs = Vector{Float64}(undef, len)
    KLs = Vector{Float64}(undef, len)
    err_ms = Vector{Float64}(undef, len)
    err_covs = Vector{Float64}(undef, len)
    err_logs = Vector{Float64}(undef, len)
    EDs = Vector{Float64}(undef, len)

    # progress bar
    prog_bar = ProgressMeter.Progress(len, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    for i = 1:len
        println("sample $i / $len")
        Args = argparse(ps[i], argparse_args...)
        # taking samples for fixed setting
        el[i], D = MCMCsampler(mcmc, Args; n_samples)
        KSDs[i] = ksd_est(D, grd)
        KLs[i] = kl_est(D, kl_args...)
        err_ms[i], err_covs[i], err_logs[i] = rel_err(rel_args..., D)
        EDs[i] = energy_dist(rel_args..., D)
        # update progress bar
        ProgressMeter.next!(prog_bar)
    end
    return el, KLs, err_ms, err_covs, err_logs, EDs, KSDs
end