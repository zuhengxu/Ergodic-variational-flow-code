using PDMats

@with_kw struct NEOobj
    d::Int64
    N_steps::Int64
    # target
    logp::Function
    ∇logp::Function
    # damp coef
    γ::Real = 1.0
    # stepsize and inverse_massmatrix
    ϵ::Real = 0.2 
    invMass::PDMats.AbstractPDMat = PDMat(I(d))
    # reference distribution
    q0_sampler::Function
    logq0::Function # this need to be lpdf of MF gaussian
end

# damped Hamiltonian transformation
# q: position; p: momentum (same notation as the paper Eq.(10)  https://openreview.net/pdf?id=76tTYokjtG)
function forward_onestep(o::NEOobj, q, p)
    d, γ, ϵ, invM, ∇logp = o.d, o.γ, o.ϵ, o.invMass, o.∇logp  
    # symplectic Euler
    pn = exp(-ϵ*γ) .* p .+ ϵ.*∇logp(q)
    qn = q .+ ϵ.* invM * pn
    return qn, pn
end

function reverse_onestep(o::NEOobj, q, p)
    d, γ, ϵ, invM, ∇logp = o.d, o.γ, o.ϵ, o.invMass, o.∇logp  
    # symplectic Euler
    qn = q .- ϵ .* invM * p
    pn = exp(ϵ*γ) .* (p .- ϵ.*∇logp(qn))
    return qn, pn
end

# lpdf of augmented reference distribution
function logq_joint(o::NEOobj, q, p)
    lpdf_mom = logpdf(MvNormalCanon(zeros(o.d), zeros(o.d), o.invMass), p)
    return o.logq0(q) .+ lpdf_mom
end
function logp_joint(o::NEOobj, q, p)
    lpdf_mom = logpdf(MvNormalCanon(zeros(o.d), zeros(o.d), o.invMass), p)
    return o.logp(q) .+ lpdf_mom
end


# importance weights, forward samples, ...
function run_single_traj(o::NEOobj, q0, p0)
    d, K, γ, ϵ = o.d, o.N_steps, o.γ, o.ϵ  
    q, p = copy(q0), copy(p0)
    # instantiate array to store
    T = zeros(2K-1, d)
    M = zeros(2K-1, d)
    logq0s = zeros(2K-1)
    logps = zeros(K)
    T[K, :] .= q0 
    M[K, :] .= p0
    logps[1] = logp_joint(o, q0, p0)
    logq0s[K] = logq_joint(o, q0, p0)

    # K-1 forward steps
    for i in 1:K-1
        q0, p0 = forward_onestep(o, q0, p0)
        lpdf = logq_joint(o, q0, p0)
        lpdf_p = logp_joint(o, q0, p0)
        T[K+i,:] .= q0
        M[K+i,:] .= p0
        logq0s[K+i] = lpdf
        logps[i+1] = lpdf_p
    end
    # K-1 backward steps 
    for j in 1:K-1
        q, p = reverse_onestep(o, q, p)
        lpdf = logq_joint(o, q, p)
        T[K-j,:] .= q
        M[K-j,:] .= p
        logq0s[K-j] = lpdf
    end
    # logJs = γ * ϵ * d .* [-(K-1):K-1 ;] 
    L = logq0s .- γ * ϵ * d .* [-K+1:K-1 ;]
    num = L[K:end]    
    denom = logsumexp_sweep(L, K)
    ISweights = exp.(num.-denom)
    Z, Ws_traj = evidence_traj(ISweights,logps, logq0s[K:end]) 
    return Z, ISweights, Ws_traj, logps, logq0s[K:end], T[K:end,:], M[K:end,:]
end

#normalization constant via one trajectory
function evidence_traj(Ws::Vector{T}, logps::Vector{T}, logqs::Vector{T}) where {T<:Real}
    @assert (length(Ws) == length(logps) ==length(logqs))
    Ws_traj = exp.(logps .- logqs) .* Ws
    return sum(Ws_traj), Ws_traj
end

#normalization constant via all trajectories
# and only save the one with largest weight from one traj
function run_all_traj(o::NEOobj, N_trajs::Int64)
    rv = MvNormalCanon(zeros(o.d), zeros(o.d), o.invMass)
    Zs = zeros(N_trajs)
    Ts = zeros(N_trajs, o.d) 
    Ms = zeros(N_trajs, o.d)
    Ws_traj = zeros(N_trajs, o.N_steps)

    @threads for i in  1:N_trajs
        q0, p0 = o.q0_sampler(), rand(rv)
        Z, _, ws, _, _, T, M = run_single_traj(o, q0, p0) 
        Zs[i] = Z
        idx = wsample(ws)
        Ws_traj[i, :] .= ws
        Ts[i, :] .= T[idx, :]
        Ms[i, :] .= M[idx, :]
    end
    return Zs, Ws_traj, Ts, Ms
end



