using TickTock, JLD
include("model_2d.jl")
include("../../inference/NF/nf.jl")
include("../common/timing.jl")



function single_nf(logp_joint, logq_joint;
                niter = 100000, elbo_size = 10, nsamples::Int64 = 2000)
        Random.seed!(1)
        # q0 = MvNormal(zeros(4), ones(4))
        q0 = MvNormal(vcat(μ, [0.0,0.0]), diagm(vcat(D.^2.0, [1.0, 1.0])))
        F = PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)
        flow = transformed(q0, F)

        tick()
        _, el, ps = nf(flow, logp_joint, logq_joint, niter; elbo_size = elbo_size)
        nf_train_time = tok()
        # evaluate elbo
        el_nf = nf_ELBO(flow, logp_joint, logq_joint; elbo_size = nsamples)

        t_nf = rand(flow, 5000)[1:2, :]
        # scatter(t_nf[1, :], t_nf[2, :])
        
        #############
        # saving result
        ##############
        file_path = joinpath("result/", string("nf",".jld"))
        JLD.save(file_path, "train_time", nf_train_time, "elbo", el_nf, "scatter", t_nf)

end

# getting sample timing/ELBO/ training time vs number layers
function running_nf(logp_joint, logq_joint; 
                    niter = 500000, elbo_size = 10, nsamples::Int64 = 2000, n_run =100)
    
        T_train = zeros(4)
        T_sample= zeros(n_run, 4)
        ELBO = zeros(4)
        q0 = MvNormal(vcat(μ, [0.0,0.0]), diagm(vcat(D.^2.0, [1.0, 1.0])))
        # q0 = MvNormal(zeros(4), ones(4))
        
        F1 = PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)∘PlanarLayer(4)
        F2 = F1∘F1
        F3 = F2∘F2
        F4 = F3∘F3∘F2 

    for (i,f) in zip(1:4, (F1, F2, F3, F4))        
        Random.seed!(i)
        flow = transformed(q0, f)
        T_sample[:, i] .= noob_timing(rand, flow; n_run = n_run)

        tick()
        _, el, ps = nf(flow, logp_joint, logq_joint, niter; elbo_size = elbo_size)
        T_train[i] = tok()

        ELBO[i] = nf_ELBO(flow, logp_joint, logq_joint; elbo_size = nsamples)
        
        # T_nf = Matrix(rand(flow, nsamples)[1:2, :]')
        # m_nf[:,:,i] .= running_mean(T_nf)
        # v_nf[:,:,i] .= running_var(T_nf)
    end
    file_path = joinpath("result/", string("NF_layer",".jld"))
    JLD.save(file_path, "n_layers", [5, 10, 20, 50], "T_train", T_train, 
                        "T_sample", T_sample, "elbo", ELBO)
end







#############33
# runining
###############
n_lfrg = 80
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer) 

niter = 100000

# joint target and joint init
logp_joint(x) = o.logp(x[1:2]) + o.lpdf_mom(x[3:4])
μ_joint = vcat(μ, [0.0,0.0])
D_joint =vcat(D , [1.0, 1.0])
logq_joint(x) =  -0.5*4*log(2π) - sum(log, abs.(D_joint)) - 0.5*sum(abs2, (x.-μ_joint)./(D_joint .+ 1e-8))
# logq_joint(x) =  -0.5*4*log(2π) - 0.5*sum(abs2, x)

@info "running single nf"
single_nf(logp_joint, logq_joint)

@info "running more nf"
running_nf(logp_joint, logq_joint)






