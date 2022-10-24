include("model_2d.jl")
include("../../inference/MCMC/adapt_NUTS.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/MCMC/hmc.jl")
# include("../../inference/MCMC/ess.jl")
include("../../inference/NF/nf.jl")
include("../common/timing.jl")
include("../common/result.jl")


function running_convergence(o::HamFlow, a::HF_params; 
                            nsamples::Int64 = 50000, n_mcmc::Int64 = 5000, n_trials::Int64 = 10)
    
    m_nuts = zeros(nsamples, 2, n_trials)
    m_hmc= zeros(nsamples, 2, n_trials)
    m_erg= zeros(nsamples, 2, n_trials)
    v_nuts= zeros(nsamples, 2, n_trials)
    v_hmc= zeros(nsamples, 2, n_trials)
    v_erg= zeros(nsamples, 2, n_trials)
               
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
        v_nuts[:,:,i] .= running_var(T_nuts)
        v_hmc[:,:,i] .=  running_var(T_hmc)
        v_erg[:,:,i] .=  running_var(T_erg)
    end
    file_path = joinpath("result/", string("running",".jld"))
    JLD.save(file_path, "m_nuts", m_nuts, "v_nuts", v_nuts,
                        "m_hmc", m_hmc, "v_hmc", v_hmc, 
                        "m_erg", m_erg, "v_erg", v_erg)
end

function  running_nf(flow, logp_joint, logq_joint; 
                    niter = 100000, elbo_size = 10, nsamples::Int64 = 50000, n_trials::Int64 = 10)
    m_nf = zeros(nsamples, 2, n_trials)
    v_nf= zeros(nsamples, 2, n_trials)

    @threads for i in 1:n_trials        
        Random.seed!(i)
        flow = transformed(q0, F)
        _, el, ps = nf(flow, logp_joint, logq_joint, niter; elbo_size = elbo_size)
        T_nf = Matrix(rand(flow, nsamples)[1:2, :]')
        m_nf[:,:,i] .= running_mean(T_nf)
        v_nf[:,:,i] .= running_var(T_nf)
    end
    file_path = joinpath("result/", string("running_nf",".jld"))
    JLD.save(file_path, "m_nf", m_nf, "v_nf", v_nf)
end
###########################
# running average
###########################
n_lfrg = 60
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std, 
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
a = ErgFlow.HF_params(0.0035*ones(d), μ, D)

running_convergence(o,a)


# # joint target and joint init
logp_joint(x) = o.logp(x[1:2]) + o.lpdf_mom(x[3:4])
μ_joint = vcat(μ, [0.0,0.0])
D_joint =vcat(D , [1.0, 1.0])
logq_joint(x) =  -0.5*4*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))

# q0 = MvNormal(zeros(4), ones(4))
q0 = MvNormal(vcat(μ, [0.0,0.0]), diagm(vcat(D.^2.0, [1.0, 1.0])))
F = PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)
flow = transformed(q0, F)

running_nf(flow, logp_joint, logq_joint)