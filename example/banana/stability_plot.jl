include("model_2d.jl")
include("../common/plotting.jl")
include("../common/result.jl")
include("../common/error.jl")

lap = JLD.load("result/stab_lap.jld")
gauss = JLD.load("result/stab_norm.jld")
logis =  JLD.load("result/stab_log.jld")

plot(lap["Ns"], vec(median(lap["fwd_err"], dims=2)), ribbon = get_percentiles(lap["fwd_err"]), lw = 3, label = "Lap Fwd", xlabel = "#refreshments", ylabel = "error", title = "Banana", 
    xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, legend=:outertopright) 
plot!(lap["Ns"], vec(median(lap["bwd_err"], dims=2)), ribbon = get_percentiles(lap["bwd_err"]), lw = 3, label = "Lap Bwd") 
plot!(lap["Ns"], vec(median(logis["fwd_err"], dims=2)), ribbon = get_percentiles(logis["fwd_err"]), lw = 3, label = "logis Fwd", xlabel = "#refreshments", ylabel = "error", title = "Banana", 
    xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, legend = :outertopright) 
plot!(lap["Ns"], vec(median(logis["bwd_err"], dims=2)), ribbon = get_percentiles(logis["bwd_err"]), lw = 3, label = "logis Bwd") 
plot!(lap["Ns"], vec(median(gauss["fwd_err"], dims=2)), ribbon = get_percentiles(gauss["fwd_err"]), lw = 3, label = "Gauss Fwd") 
plot!(lap["Ns"], vec(median(gauss["bwd_err"], dims=2)), ribbon = get_percentiles(gauss["bwd_err"]), lw = 3, label = "Gauss Bwd") 
savefig("figure/banana_stability.png")