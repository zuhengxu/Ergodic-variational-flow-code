using TickTock, JLD, PDMats
import PlotlyJS as pjs
include("model_2d.jl")
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

# optimal setting---fixed
o_neo = NEO.NEOobj(d = d, 
                N_steps = 10,  
                logp = logp, 
                ∇logp = ∇logp, 
                γ = 0.2, 
                ϵ = 0.5, 
                invMass = PDMat(I(d)), 
                q0_sampler = q0_sampler,
                logq0 =logq0)        


# ##################
# # density and scatter 
# ####################
# X = [-30.001:0.5:30 ;]
# Y = [-30.001:0.5:30 ;]

# layout = pjs.Layout(
#     width=500, height=500,
#     scene = pjs.attr(
#         xaxis = pjs.attr(showticklabels=false, visible=false),
#         yaxis = pjs.attr(showticklabels=false, visible=false),
#         zaxis = pjs.attr(showticklabels=false, visible=false, range = [-5000, 0]),
#     ),
#     margin=pjs.attr(l=0, r=0, b=0, t=0, pad=0),
#     colorscale = "Vird"
# )


# T = lpdf_neo_save(o_neo, X, Y; res_dir = "result/", res_name = "lpdf_neo.jld")

# p_est = pjs.plot(pjs.surface(z=T, x=X, y=Y, cauto = false, cmax = 0, cmin = -5000, showscale=false), layout)
# pjs.savefig(p_est, joinpath("figure/","lpdf_neo.png"))




##################
# timing
####################
neo_timing(o_neo; nchains = 10, mcmciters = 10, nrun = 100, res_dir = "result/", res_name = "neo_time.jld2")
neo_ess_time(o_neo; nchains = 10, mcmciters = 5000, nadapt = 0, n_run = 10, Adapt = false, res_dir = "result/", res_name = "ess_neo.jld2")