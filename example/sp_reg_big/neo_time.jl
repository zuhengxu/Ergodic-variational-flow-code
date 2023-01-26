using TickTock, JLD, PDMats
import PlotlyJS as pjs
include("model.jl")
include("../common/neo_run.jl")
include("../../inference/SVI/svi.jl")
include("../common/result.jl")



##################
# setting 
####################

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]

logq0(x) = logq(x, μ, D)
q0_sampler() = randn(d).*D .+ μ

# optimal setting---adapt
o_neo = NEO.NEOobj(d = d, 
                N_steps = 10,  
                logp = logp, 
                ∇logp = ∇logp, 
                γ = 0.2, 
                ϵ = 0.2, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)        


##################
# timing
####################
# neo_timing(o_neo; nchains = 10, mcmciters = 10, nrun = 100, res_dir = "result/", res_name = "neo_time.jld2")
neo_ess_time(o_neo; nchains = 10, mcmciters = 5000, nadapt = 5000, n_run = 10, Adapt = true, res_dir = "result/", res_name = "ess_neo.jld2")