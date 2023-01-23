using TickTock, JLD, PDMats
include("model_2d.jl")
include("../common/timing.jl")
include("../../inference/NEO/NEO.jl")
include("../../inference/SVI/svi.jl")

using JLD
using Base.Threads: @threads 

using Distributions, Random
###########3
#  setting
###########
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
# n_lfrg = 200
# o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
#         randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
#         ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)


logq0(x) = logq(x, μ, D)
q0_sampler() = randn(d).*D .+ μ
# q0_sampler() = vec(q0_sampler(1))

o_neo = NEO.NEOobj(d =2, 
                N_steps = 20,  
                logp = logp, 
                ∇logp = ∇logp, 
                γ = 0.5, 
                ϵ = 0.2, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)        

# time persample 
q0,p0 = q0_sampler(), randn(2)
Zn, ISws, Ws_traj,logps,logqs,T, M= NEO.run_single_traj(o_neo, q0, p0)
NEO.run_all_traj(o_neo, 10)
T, M, o_new = NEO.neomcmc(o_neo, 10, 50000; Adapt = true)

MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
x = -20.:.1:20
y = -15:.1:30
f = (x,y) -> exp(logp([x, y]))
p1 = contour(x, y, f, colorbar = false, title = "Banana")
scatter!(T[:, 1], T[:,2])

scatter(T[:, 1], T[:,2])
# ess time
o_ad = NEO.NEOadaptation(o_neo; n_adapts=100000)
Zn, ISws, Ws_traj,logps,logqs,T, M= NEO.run_single_traj(o_ad, q0, p0)
ϵ, invMass = NEO.HMC_get_adapt(q0, 0.65, o.logp, o.∇logp, 50000; nleapfrog = o_neo.N_steps)


