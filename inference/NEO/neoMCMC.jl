include("hmc_adapt.jl")
using Accessors

function NEOadaptation(o::NEOobj; n_adapts = 10000, target_acc = 0.7, verbose = true)
    # using adaptation from HMC
    q0 = o.q0_sampler()
    stepsize, invM = HMC_get_adapt(q0, target_acc, o.logp, o.∇logp, n_adapts; nleapfrog = o.N_steps, verbose = verbose)

    o = @set o.ϵ = stepsize
    o = @set o.invMass = PDMat(diagm(invM))
    return o
end


#########
# NEO-MCMC
############

function neomcmc_update!(o::NEOobj, N_trajs::Int64, q0, p0, rv, Zs, T0s, M0s, Ts, Ms)
    # Zs = zeros(N_trajs)
    # Ts = zeros(N_trajs, o.d) 
    # Ms = zeros(N_trajs, o.d)
    # T0s = zeros(N_trajs, o.d) 
    # M0s = zeros(N_trajs, o.d)

    @threads for i in  1:N_trajs
        # set Xn1 = Yn-1 (step1: 1.)
        q0, p0 = i > 1 ? (o.q0_sampler(), rand(rv)) : (q0, p0)
        T0s[i, :] .= q0
        M0s[i, :] .= p0

        Z, _, ws, _, _, T, M = run_single_traj(o, q0, p0) 
        Zs[i] = Z
        idx = wsample(ws)
        # Ws_traj[i, :] .= ws
        Ts[i, :] .= T[idx, :]
        Ms[i, :] .= M[idx, :]
    end
    idx_chain = wsample(Zs)
    return T0s[idx_chain,:], M0s[idx_chain,:], Ts[idx_chain,:], Ms[idx_chain,:]
end

function neomcmc(o::NEOobj, N_trajs::Int64, n_samples::Int64; 
                Adapt::Bool = true, n_adapts::Int64 = n_samples, adp_acc = 0.7, adp_verbose=true)
    if Adapt
        o = NEOadaptation(o; n_adapts = n_adapts, target_acc=adp_acc, verbose = adp_verbose)
    end
    Zs = zeros(N_trajs)
    Ts = zeros(N_trajs, o.d) 
    Ms = zeros(N_trajs, o.d)
    T0s = zeros(N_trajs, o.d) 
    M0s = zeros(N_trajs, o.d)
    
    T_samples = zeros(n_samples, o.d)
    M_samples = zeros(n_samples, o.d)

    rv = MvNormalCanon(zeros(o.d), zeros(o.d), o.invMass)
    q0, p0 = o.q0_sampler(), rand(rv)
    # run
    pm = ProgressMeter.Progress(n_samples, desc="NEOmcmc", barlen=31)
    for i in 1:n_samples
        q0, p0, q, p = neomcmc_update!(o, N_trajs, q0, p0, rv, Zs, T0s, M0s, Ts, Ms)
        T_samples[i, :] .= q
        M_samples[i, :] .= p
        ProgressMeter.next!(pm)
    end
    return T_samples, M_samples, o
end
