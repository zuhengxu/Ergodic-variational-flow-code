include("model.jl")
include("neo.jl")
include("../../inference/MCMC/adapt_NUTS.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/MCMC/hmc.jl")
# include("../../inference/MCMC/ess.jl")
# include("../../inference/NF/nf.jl")
include("../common/timing.jl")
include("../common/result.jl")


function running_convergence(o::HamFlow, a::HF_params; 
                            nsamples::Int64 = 50000, n_mcmc::Int64 = 5000, n_trials::Int64 = 10)
    
    m_nuts = zeros(nsamples, d, n_trials)
    m_nuts_ad = zeros(nsamples, d, n_trials)
    m_hmc= zeros(nsamples, d, n_trials)
    m_erg= zeros(nsamples, d, n_trials)
    m_erg_single = zeros(nsamples, d, n_trials)
    v_nuts= zeros(nsamples, d, n_trials)
    v_nuts_ad= zeros(nsamples, d, n_trials)
    v_hmc= zeros(nsamples, d, n_trials)
    v_erg= zeros(nsamples, d, n_trials)
    v_erg_single = zeros(nsamples, d, n_trials)
               
    @threads for i in 1: n_trials
        Random.seed!(i)
        z0= randn(o.d) .* a.D .+ a.μ
        ρ0 = o.ρ_sampler(o.d)
        u0 = rand()

        @info "sampling nuts"
        T_nuts = nuts(z0, 0.65, o.logp, o.∇logp, nsamples, 0)

        @info "sampling nuts adaptive"
        T_nuts_ad = nuts(z0, 0.65, o.logp, o.∇logp, nsamples/2, nsamples/2)

        @info "sampling hmc"
        T_hmc = hmc(z0, a.leapfrog_stepsize, 0.65, o.n_lfrg, o.logp, o.∇logp, nsamples, 0)

        @info "sampling ErgFlow"
        # T_erg, _, _ = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, nsamples)
        T_erg = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, n_mcmc, nsamples)

        @info "sampling ErgFlow single"
        # T_erg, _, _ = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, nsamples)
        T_erg_single = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, nsamples, nsamples)

        m_nuts[:,:,i] .= running_mean(T_nuts)
        m_nuts_ad[:,:,i] .= running_mean(T_nuts_ad)
        m_hmc[:,:,i] .=  running_mean(T_hmc)
        m_erg[:,:, i] .=  running_mean(T_erg)
        m_erg_single[:,:, i] .=  running_mean(T_erg_single)
        v_nuts[:,:,i] .= running_second_moment(T_nuts)
        v_nuts_ad[:,:,i] .= running_second_moment(T_nuts_ad)
        v_hmc[:,:,i] .=  running_second_moment(T_hmc)
        v_erg[:,:,i] .=  running_second_moment(T_erg)
        v_erg_single[:,:,i] .=  running_second_moment(T_erg_single)
    end
    file_path = joinpath("result/", string("running",".jld"))
    JLD.save(file_path, "m_nuts", m_nuts, "v_nuts", v_nuts,
                        "m_nuts_ad", m_nuts_ad, "v_nuts_ad", v_nuts_ad,
                        "m_hmc", m_hmc, "v_hmc", v_hmc, 
                        "m_erg", m_erg, "v_erg", v_erg,
                        "m_erg_single", m_erg_single, "v_erg_single", v_erg_single)
end

###########################
# running average
###########################
n_lfrg = 50
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
a = ErgFlow.HF_params(3e-5*ones(d), μ, D)

running_convergence(o,a)