include("model_2d.jl")
include("../../inference/MCMC/adapt_NUTS.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/MCMC/hmc.jl")
# include("../../inference/MCMC/ess.jl")
include("../../inference/NF/nf.jl")
include("../common/timing.jl")
include("../common/result.jl")




function running_convergence(o::HamFlow, a::HF_params; 
                            nsamples::Int64 = 50000, n_mcmc::Int64 = 2000, n_trials::Int64 = 10)
    
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
########################3
# n_lfrg = 80
# o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
#         ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std,ErgFlow.pdf_laplace_std, 
#         ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 
# a = ErgFlow.HF_params(0.003*ones(d), μ, D)

# running_convergence(o,a)

# # # joint target and joint init
# logp_joint(x) = o.logp(x[1:2]) + o.lpdf_mom(x[3:4])
# μ_joint = vcat(μ, [0.0,0.0])
# D_joint =vcat(D , [1.0, 1.0])
# logq_joint(x) =  -0.5*4*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))

# # q0 = MvNormal(zeros(4), ones(4))
# q0 = MvNormal(vcat(μ, [0.0,0.0]), diagm(vcat(D.^2.0, [1.0, 1.0])))
# F = PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)
# flow = transformed(q0, F)

# running_nf(flow, logp_joint, logq_joint)



###########################
# comparing iid and trajectory average
########################3
include("../common/plotting.jl")
function var_compare(o::HamFlow, a::HF_params, f::Function; nsamples::Int64 = 50000, n_mcmc::Int64 = 1000, n_trials::Int64 = 10)
    
    nsample_iid = 2*Int(nsamples/n_mcmc)
    m_iid= zeros(nsample_iid, n_trials)
    m_erg= zeros(nsamples, n_trials)
    Ns = zeros(nsample_iid, n_trials)
               
    @threads for i in 1: n_trials
        Random.seed!(i)

        @info "sampling iid"
        T, M, U, steps = ErgFlow.Sampler_with_ind(o, a, pseudo_refresh_coord, n_mcmc, nsample_iid)

        @info "sampling traj"
        T_erg, M_erg, U_erg = ErgFlow.flow_sampler_all(o, a, pseudo_refresh_coord, n_mcmc, nsamples)

        Ns[:, i] .= steps
        m_iid[:,i] .= running_average(f, T, M, U)
        m_erg[:,i] .= running_average(f, T_erg, M_erg, U_erg)
    end
    return m_iid, m_erg, Ns
end

n_lfrg = 80
o = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
a = ErgFlow.HF_params(0.003*ones(d), μ, D)

f(x) = sum(abs, x)
nsample = 50000
n_mcmc = 2000
m_iid, m_erg, steps = var_compare(o, a, f; nsamples = nsample, n_mcmc = n_mcmc, n_trials = 100)
JLD.save("result/var_compare.jld", "m_iid", m_iid, "m_erg", m_erg, "steps", steps)


m_iid, m_erg, steps = JLD.load("result/var_compare.jld", "m_iid", "m_erg", "steps")
iters = cumsum(steps, dims = 1)
ts = time_range(iters')

p1 = plot(ts, time_mean(m_iid', iters'), ribbon = time_std(m_iid', iters'),lw = 4, label = "iid", xticks = [0:10000:nsample ;])
    plot!(n_mcmc:n_mcmc:nsample, mean(m_erg, dims = 2)[n_mcmc:n_mcmc:end], ribbon = std(m_erg, dims=2)[n_mcmc:n_mcmc:end], lw = 4,label = "Traj. ave.", xticks = [0:10000:nsample ;])
    # hline!([0.0],  linestyle=:dash, lw = 2,color =:black,label = "E[f]")
    plot!(title = "Warped Gauss", xlabel = "#Refreshments", ylabel = "")
    plot!(size = (1800,1500), xtickfontsize = 40, ytickfontsize =50,margin=10Plots.mm, guidefontsize= 50, xrotation = 15,
    titlefontsize = 60, legend=:top, legendfontsize = 50 )

savefig(p1, "figure/warp_var_compare.png")



