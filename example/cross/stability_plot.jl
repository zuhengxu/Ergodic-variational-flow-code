include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")
using StatsBase

lap = JLD.load("result/stab_lap.jld")
gauss = JLD.load("result/stab_norm.jld")

function get_percentiles(dat; p1=25, p2=75)
    dat = Matrix(dat')
    n = size(dat,2)
    median_dat = vec(median(dat, dims=1))

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        plow[i] = median_dat[i] - percentile(vec(dat[:,i]), p1)
        phigh[i] = percentile(vec(dat[:,i]), p2) - median_dat[i]
    end

    return plow, phigh
end

plot(lap["Ns"], vec(median(lap["fwd_err"], dims=2)), ribbon = get_percentiles(lap["fwd_err"]), lw = 3, label = "Lap Fwd", xlabel = "#refreshments", ylabel = "error", title = "Cross", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, margin=5Plots.mm, legend=:outertopright) 
plot!(lap["Ns"], vec(median(lap["bwd_err"], dims=2)), ribbon = get_percentiles(lap["bwd_err"]), lw = 3, label = "Lap Bwd") 
plot!(lap["Ns"], vec(median(gauss["fwd_err"], dims=2)), ribbon = get_percentiles(gauss["fwd_err"]), lw = 3, label = "Gauss Fwd") 
plot!(lap["Ns"], vec(median(gauss["bwd_err"], dims=2)), ribbon = get_percentiles(gauss["bwd_err"]), lw = 3, label = "Gauss Bwd") 
savefig("figure/cross_stability.png")