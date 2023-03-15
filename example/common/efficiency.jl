include("../../inference/MCMC/adapt_NUTS.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/MCMC/hmc.jl")
include("../../inference/MCMC/ess.jl")
include("../common/timing.jl")



function run_time_per_sample(o::HamFlow, a::HF_params, refresh::Function; n_run = 100, n_mcmc = 2000, seed = 2022)
    Random.seed!(seed)
    # n_run = 100
    # n_mcmc = 2000
    
    z0= randn(d) .* a.D .+ a.μ
    ρ0 = o.ρ_sampler(o.d)
    u0 = rand()

    @info "timing sample generation NUTS"
    time_sample_nuts = noob_timing(nuts, z0, 0.7, o.logp, o.∇logp, 2, 0; n_run =n_run) / 2

    @info "timing sample generation hmc"
    time_sample_hmc = noob_timing(hmc, z0, a.leapfrog_stepsize, 0.7, n_lfrg, o.logp, o.∇logp, 1, 0; n_run = n_run)

    @info "timing sample geneneration ergflow IID"
    time_sample_erg_iid = noob_timing(flow_fwd, o, a.leapfrog_stepsize, refresh, z0, ρ0, u0, n_mcmc; n_run = n_run)

    @info "timing sample generation ergflow all"
    time_sample_erg_single = noob_timing(flow_fwd_trace, o, a.leapfrog_stepsize, refresh, z0, ρ0, u0, n_mcmc; n_run = n_run)./n_mcmc 

    # q0 = MvNormal(zeros(4), ones(4))
    # F = PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)
    # flow = transformed(q0, F)
    # @info "timing sample generation NF"
    # time_sample_nf = noob_timing(rand, flow; n_run = n_run)

    @info "save time per sample"
    file_path = joinpath("result/", string("timing_per_sample",".jld"))
    JLD.save(file_path, "time_sample_nuts", time_sample_nuts, 
                        "time_sample_hmc", time_sample_hmc, 
                        "time_sample_erg_iid", time_sample_erg_iid, "time_sample_erg_single", time_sample_erg_single)

end


# ESS per time unit
function ess_time(o::HamFlow, a::HF_params, refresh::Function; 
                    num_trials::Int = 10, nsamples::Int = 2000, n_mcmc::Int = 2000, nadapt::Int = 2000)
    
    t_nuts, ess_nuts, ess_time_nuts= zeros(num_trials), zeros(num_trials), zeros(num_trials)
    t_nuts_ad, ess_nuts_ad, ess_time_nuts_ad= zeros(num_trials), zeros(num_trials), zeros(num_trials)
    t_hmc, ess_hmc, ess_time_hmc= zeros(num_trials), zeros(num_trials), zeros(num_trials)
    t_erg_iid, ess_erg_iid, ess_time_erg_iid= zeros(num_trials), zeros(num_trials), zeros(num_trials)
    t_erg_single, ess_erg_single, ess_time_erg_single = zeros(num_trials), zeros(num_trials), zeros(num_trials)

    for i in 1: num_trials
        @info "$i/$num_trials"

        Random.seed!(i)
                
        # nsamples = 2000
        z0= randn(d) .* a.D .+ a.μ
        ρ0 = o.ρ_sampler(o.d)
        u0 = rand()
        # n_mcmc = 2000

        @info "ESS/time NUTS"
        tick()
        T_nuts = nuts(z0, 0.7, o.logp, o.∇logp, nsamples, 0)
        t_nuts[i] = tok()
        ess_nuts[i] = ess(T_nuts)
        ess_time_nuts[i] = ess_nuts[i]/t_nuts[i]
        println(ess_time_nuts[i])
        
        @info "ESS/time NUTS adapt"
        tick()
        T_nuts_ad = nuts(z0, 0.7, o.logp, o.∇logp, nsamples, nadapt)
        t_nuts_ad[i] = tok()
        ess_nuts_ad[i] = ess(T_nuts_ad)
        ess_time_nuts_ad[i] = ess_nuts_ad[i]/t_nuts_ad[i]
        println(ess_time_nuts_ad[i])

        @info "ESS/time NUTS"
        tick()
        T_nuts = adapt_nuts(z0, 0.7, o.logp, o.∇logp, nsamples, nadapt)
        t_nuts[i] = tok()
        ess_nuts[i] = ess(T_nuts)
        ess_time_nuts[i] = ess_nuts[i]/t_nuts[i]
        
        @info "ESS/time HMC"
        tick()
        T_hmc = hmc(z0, a.leapfrog_stepsize, 0.7, o.n_lfrg, o.logp, o.∇logp, nsamples, nadapt)
        t_hmc[i] = tok()
        ess_hmc[i] = ess(T_hmc)
        ess_time_hmc[i] = ess_hmc[i]/t_hmc[i]

        @info "ESS/time ErgFlow single trace"
        tick()
        T_single, _, _ = flow_sampler(o, a, refresh, z0, ρ0, u0, nsamples)
        t_erg_single[i] = tok()
        ess_erg_single[i] = ess(T_single)
        ess_time_erg_single[i] = ess_erg_single[i]/t_erg_single[i]

        @info "ESS/time ErgFlow iid"
        tick()
        T_iid = zeros(nsamples, o.d)
        Sampler!(T_iid, o, a, refresh, n_mcmc, nsamples)
        t_erg_iid[i] = tok()
        ess_erg_iid[i] = ess(T_iid)
        ess_time_erg_iid[i] = ess_erg_iid[i] /t_erg_iid[i]
            
    end
    @info "save ESS per time unit plot"
    file_path = joinpath("result/", string("ESS",".jld"))
    JLD.save(file_path, "t_nuts", t_nuts, "ess_nuts", ess_nuts, "ess_time_nuts", ess_time_nuts, 
                        "t_nuts_ad", t_nuts_ad, "ess_nuts_ad", ess_nuts_ad, "ess_time_nuts_ad", ess_time_nuts_ad,
                        "t_hmc", t_hmc, "ess_hmc", ess_hmc, "ess_time_hmc", ess_time_hmc, 
                        "t_erg_iid", t_erg_iid, "ess_erg_iid", ess_erg_iid, "ess_time_erg_iid", ess_time_erg_iid,
                        "t_erg_single", t_erg_single, "ess_erg_single", ess_erg_single, "ess_time_erg_single", ess_time_erg_single)
end
