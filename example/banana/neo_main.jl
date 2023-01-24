using TickTock, JLD, PDMats
include("model_2d.jl")
include("../common/neo_run.jl")
include("../../inference/SVI/svi.jl")





###########3
#  setting
###########
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

# o_neo = NEO.NEOobj(d=2, 
#                 N_steps = 20,  
#                 logp = logp, 
#                 ∇logp = ∇logp, 
#                 γ = 0.5, 
#                 ϵ = 0.2, 
#                 invMass = PDMat(I(d)), 
#                 q0_sampler = q0_sampler,
#                 logq0 =logq0)        

logq0(x) = logq(x, μ, D)
q0_sampler() = randn(d).*D .+ μ

neo_fixed_run(d, logp, ∇logp, q0_sampler, logq0; 
            γs = [0.2, 0.5, 1.0], Ks = [10, 20], ϵs =[0.2, 0.5, 1.0], mcmciters = 20000, nchains = [10], ntrials=3, 
            ntests = 20, nKSD = 5000, 
            res_dir = "result/", csv_name = "neo_fix.csv", jld_name = "neo_fix.jld2")

neo_adaptation_run(d, logp, ∇logp, q0_sampler, logq0; 
                γs= [0.2, 0.5, 1.0], Ks= [10, 20], nchains = [10], mcmciters = 20000, nadapt = 20000, ntrials = 2, 
                ntests = 20, nKSD = 5000, 
                res_dir = "result/", csv_name = "neo_adp.csv", jld_name = "neo_adp.jld2")         

