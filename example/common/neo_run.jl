using TickTock, JLD2, Random, DataFrames, CSV 
using PDMats
using Base:Threads
include("../../inference/util/ksd.jl")
include("../../inference/NEO/NEO.jl")
include("../../inference/MCMC/ess.jl")
include("../common/timing.jl")


function neo_fixed_run(d, logp, ∇logp, q0_sampler, logq0; 
                        γs, Ks, ϵs, mcmciters = 50000, nchains = [15], ntrials=3, ntests = 20, 
                        res_dir = "result/", csv_name = "neo_fix.csv", jld_name = "neo_fix.jld2")
    # invMass is just gonna be I
    # need to store the optimal setting in a table

    # initiate an empty dataframe
    df = DataFrame()
    # different settings
    for γ in γs, K in Ks, e in ϵs, c in nchains
        # construct NEO object for each setting
        o = NEO.NEOobj(d = d,  
                N_steps = K,  
                logp = logp, 
                ∇logp = ∇logp, 
                γ = γ, 
                ϵ = e, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)

        # repeat each setting 
        for i in 1:ntrials 
            Random.seed!(i)
            tick()
            T, M, o_new = NEO.neomcmc(o, c, mcmciters; Adapt = false)
            time = tok()
            Ztest, _,_,_ = NEO.run_all_traj(o_new, ntests)
            NaNratio = sum(isnan.(Ztest))/ntests 
            # compute ksd for the last 5000 samples as criterion
            ksd_est = ksd(T[mcmciters-4999:end , :], ∇logp)            
            push!(df, (id = i, gamma = γ, Nsteps = K, stepsize = e, Nchians = c, KSD = ksd_est, run_time = time, NaNratio = NaNratio))
            @info "id = $i, gamma = $γ, Nsteps = $K, stepsize = $e, Nchians = $c, KSD = $ksd_est, run_time = $time, NaNratio = $NaNratio" 
        end
    end
    path = joinpath(res_dir, csv_name) 
    CSV.write(path, df; delim = "\t")
    path = joinpath(res_dir, jld_name) 
    JLD2.save(path, "df", df)
end


function neo_adaptation_run( d, logp, ∇logp, q0_sampler, logq0; 
                            γs, Ks, nchains = [15], mcmciters = 50000, nadapt = 50000, ntrials = 3, ntests = 20,
                            res_dir = "result/", csv_name = "neo_adp.csv", jld_name = "neo_adp.jld2")
    # invMass and ϵ will be adapted
    # need to store the optimal setting in a table
    
    # initiate an empty dataframe
    df = DataFrame()
    # different settings
    for γ in γs, K in Ks, c in nchains
        # construct NEO object for each setting
        o = NEO.NEOobj(d = d,  
                N_steps = K,  
                logp = logp, 
                ∇logp = ∇logp, 
                γ = γ, 
                ϵ = 0.1, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)
        # repeat each setting 
        for i in 1:ntrials 
            Random.seed!(i)
            tick()
            T, M, o_new = NEO.neomcmc(o, c, mcmciters; Adapt = true, n_adapts = nadapt)
            time = tok()
            Ztest, _,_,_ = NEO.run_all_traj(o_new, ntests)
            NaNratio = sum(isnan.(Ztest))/ntests 
            e = o_new.ϵ
            # compute ksd for the last 5000 samples as criterion
            ksd_est = ksd(T[mcmciters-4999:end , :], ∇logp)            
            push!(df, (id = i, gamma = γ, Nsteps = K, stepsize = e, Nchians = c, KSD = ksd_est, run_time = time, NaNratio = NaNratio))
            @info "id = $i, gamma = $γ, Nsteps = $K, stepsize = $e, Nchians = $c, KSD = $ksd_est, run_time = $time, NaNratio = $NaNratio" 
        end
    end
    path = joinpath(res_dir, csv_name) 
    CSV.write(path, df; delim = "\t")
    path = joinpath(res_dir, jld_name) 
    JLD2.save(path, "df", df)
end


function neo_timing(o_neo; nchains = 15, mcmciters = 100, nrun = nrun)
    # we do not include adapation time
    @info "timing sample generation for neo"
    func(o::NEO.NEOobj) = NEO.neomcmc(o_neo, nchains, mcmciters; Adapt = false)
    times = noob_timing(func, o_neo; n_run = n_run)./mcmciters 
    path = joinpath(res_dir, res_name) 
    JLD2.save(path, "times", times, "nchains", nchains, "mcmciters",100, "stepsize",o_neo.ϵ, "K", o_neo.N_steps,"gamma", o_neo.γ)
end

function neo_ess_time(o_neo; nchains = 15, mcmciters = 5000, nrun = nrun, Adapt = true, nadapt = 5000, 
                    res_dir = "result/", res_name = "ess_neo.jld2")
    t_neo, ess_neo, ess_time_neo= zeros(n_run), zeros(n_run), zeros(n_run)
    for i in 1:n_run
        @info "$i/$num_trials"
        Random.seed!(i)

        @info "ESS/time NEO"
        tick()
        T_neo, M, o_new = NEO.neomcmc(o_neo, nchains, mcmciters; Adapt = Adapt, n_adapts = nadapt)
        t_neo[i] = tok()
        ess_neo[i] = ess(T_neo)
        ess_time_neo[i] = ess_neo[i]/t_neo[i]
    end

    @info "save ESS per time unit"
    file_path = joinpath(res_dir, res_name)
    JLD2.save(file_path, "t_neo", t_neo, "ess_neo", ess_neo, "ess_time_neo", ess_time_neo) 
end
