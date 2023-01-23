using TickTock, JLD, PDMats
include("model_2d.jl")
include("../common/timing.jl")
include("../../inference/NEO/NEO.jl")
include("../../inference/SVI/svi.jl")

using JLD
using Base.Threads: @threads 


###########3
#  setting
###########
MF = JLD.load("result/mfvi.jld")
μ, D = MF["μ"], MF["D"]
n_lfrg = 200
o = ErgFlow.HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
        ErgFlow.stream, ErgFlow.mixer, ErgFlow.inv_mixer)


logq0(x) = logq(x, μ, D)
q0_sampler(n::Int64) = randn(d,n).*D .+ μ
q0_sampler() = q0_sampler(1)

o_neo = NEO.NEOobj(d =2, 
                N_steps = 10, 
                logp = logp, 
                ∇logp = ∇logp, 
                γ = 1.0, 
                ϵ = 0.2, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)        



# time persample 




# ess time