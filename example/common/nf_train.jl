using TickTock, JLD2
using Base:Threads
using DataFrames
include("../../inference/util/ksd.jl")
include("../../inference/NF/nf.jl")
include("../common/timing.jl")


# save settings/ELBO/training time/sampling time/samples/
function single_nf(logp_joint, logq_joint, μ, D, d;
                nlayers = 5, hdims= d, flow_type = "Planar",
                niter = 200000, elbo_size = 10, nelbo_est = 2000, nsamples::Int64 = 5000, n_run = 100, 
                seed = 1, file_name = flow_type*"$(nlayers).jld2" )
        
        Random.seed!(seed)
        # base distribution for NF
        q0 = MvNormal(vcat(μ, zeros(d)), diagm(vcat(D.^2.0, ones(d))))
        # specify flow type
        if flow_type == "RealNVP"
                tick() 
                flow, _, _, _ = train_rnvp(q0, logp_joint, logq_joint, 2d, niter; hdims =hdims, elbo_size = elbo_size)
                nf_train_time = tok()

        elseif flow_type ∈ ("Planar", "Radial") 
                
                if flow_type == "Planar"
                        F = ∘([PlanarLayer(2d) for i in 1:nlayers]...)
                elseif flow_type == "Radial"
                        F = ∘([RadialLayer(2d) for i in 1:nlayers]...)
                end
                flow = transformed(q0, F)
                tick()
                _, el, ps = nf(flow, logp_joint, logq_joint, niter; elbo_size = elbo_size)
                nf_train_time = tok()
        else 
                println("flow type not defined")
        end

        # est elbo
        el_nf = nf_ELBO(flow, logp_joint, logq_joint; elbo_size = nelbo_est)
        # time persample
        sampling_time = noob_timing(rand, flow; n_run = n_run)
        # collecting samples from trained nf
        t_nf = rand(flow.dist, nsamples)'[:, 1:d]
        
        #############
        # saving result
        ##############
        file_path = joinpath("result/", file_name)
        JLD2.save(file_path, "train_time", nf_train_time, "sampling_time", sampling_time, 
                "elbo", el_nf, "Samples", t_nf,  
                "type", flow_type, "nlayer", nlayers, "hdims", hdims)

end

# computing elbo for a single setting
function get_nf_elbo(logp_joint, logq_joint, μ, D, d; 
                nlayers = 5, hdims = d, flow_type = "Planar",
                niter = 200000, elbo_size = 10, nelbo_est = 2000, seed = 1)
        
        Random.seed!(seed)
        # base distribution for NF
        q0 = MvNormal(vcat(μ, zeros(d)), diagm(vcat(D.^2.0, ones(d))))

        # specify flow type
        if flow_type == "RealNVP"
                tick() 
                flow, _, _, _ = train_rnvp(q0, logp_joint, logq_joint, 2d, niter;nlayers = nlayers, hdims =hdims, elbo_size = elbo_size)
                nf_train_time = tok()

        elseif flow_type ∈ ("Planar", "Radial") 
                
                if flow_type == "Planar"
                        F = ∘([PlanarLayer(2d) for i in 1:nlayers]...)
                elseif flow_type == "Radial"
                        F = ∘([RadialLayer(2d) for i in 1:nlayers]...)
                end
                flow = transformed(q0, F)
                tick()
                _, el, ps = nf(flow, logp_joint, logq_joint, niter; elbo_size = elbo_size)
                nf_train_time = tok()
        else 
                println("flow type not defined")
        end
        
        # est elbo
        el_nf = nf_ELBO(flow, logp_joint, logq_joint; elbo_size = nelbo_est)
        @info flow_type*"_layer = $(nlayers), seed = $(seed), elbo = $(el_nf)"
        return  (elbo = el_nf, flow_type = flow_type, nlayers = nlayers, hdims = hdims, rand_seed = seed, train_time = nf_train_time)
end

# storing elbo for K trials
function tune_nf(logp_joint, logq_joint, μ, D, d; 
                nlayers = [5, 10, 20], hdims = d, flow_type = "Planar",
                niter = 200000, elbo_size = 10, nelbo_est = 2000, nrun = 5, 
                file_name = flow_type*"_tune.jld2")
    
        K = size(nlayers, 1)
        ELBO = zeros(nrun, K)
        T_train = zeros(nrun, K) 

        # running different layer setting for nruns
        # Threads.@threads 
        for ij in CartesianIndices((nrun, K)) 
                # i = num run, j = num layers
                i, j = Tuple(ij) 
                res = get_nf_elbo(logp_joint, logq_joint, μ, D, d; flow_type = flow_type, 
                                        nlayers = nlayers[j], hdims =hdims, niter = niter, elbo_size = elbo_size, nelbo_est = nelbo_est, seed = i)
                ELBO[i, j] = res.elbo
                T_train[i, j] = res.train_time
        end
        # write result into df
        df_elbo = DataFrame(ELBO, :auto)
        rename!(df_elbo, ["$(n)layers" for n in nlayers])
        df_train = DataFrame(T_train, :auto)
        rename!(df_train, ["$(n)layers" for n in nlayers])

    file_path = joinpath("result/", file_name)
    JLD2.save(file_path, "n_layers", nlayers, "hdims", hdims, "type", flow_type, "train_time", df_train, "elbo", df_elbo)
end

###################
# compute ksd using NF samples
###################
function nf_ksd(files, o; file_name = "NF_ksd.jld2", kwargs...)
        T_ksd = zeros(size(files,1))

        for i in 1:size(files,1)
                Dnf = JLD2.load("result/"*files[i])["Samples"]
                ksd_est = ksd(Dnf, o.∇logp; kwargs...)
                T_ksd[i] = ksd_est 
                println("$(files[i]) ksd =  $(ksd_est)")
        end
        file_path = joinpath("result/", file_name)
        JLD2.save(file_path, "ksd", T_ksd, "settings", files)
end