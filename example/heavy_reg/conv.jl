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
    m_hmc= zeros(nsamples, d, n_trials)
    m_erg= zeros(nsamples, d, n_trials)
    v_nuts= zeros(nsamples, d, n_trials)
    v_hmc= zeros(nsamples, d, n_trials)
    v_erg= zeros(nsamples, d, n_trials)
               
    @threads for i in 1: n_trials
        Random.seed!(i)
        z0= randn(o.d) .* a.D .+ a.μ
        ρ0 = o.ρ_sampler(o.d)
        u0 = rand()

        @info "sampling nuts"
        T_nuts = nuts(z0, 0.65, o.logp, o.∇logp, nsamples, 0)

        @info "sampling hmc"
        T_hmc = hmc(z0, a.leapfrog_stepsize, 0.65, o.n_lfrg, o.logp, o.∇logp, nsamples, 0)

        @info "sampling ErgFlow"
        # T_erg, _, _ = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, nsamples)
        T_erg = flow_sampler(o, a, pseudo_refresh_coord, z0, ρ0, u0, n_mcmc, nsamples)

        m_nuts[:,:,i] .= running_mean(T_nuts)
        m_hmc[:,:,i] .=  running_mean(T_hmc)
        m_erg[:,:, i] .=  running_mean(T_erg)
        v_nuts[:,:,i] .= running_second_moment(T_nuts)
        v_hmc[:,:,i] .=  running_second_moment(T_hmc)
        v_erg[:,:,i] .=  running_second_moment(T_erg)
    end
    file_path = joinpath("result/", string("running",".jld"))
    JLD.save(file_path, "m_nuts", m_nuts, "v_nuts", v_nuts,
                        "m_hmc", m_hmc, "v_hmc", v_hmc, 
                        "m_erg", m_erg, "v_erg", v_erg)
end

###########################
# running average
###########################
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
a = ErgFlow.HF_params(8e-4*ones(d), μ, D)

running_convergence(o,a)