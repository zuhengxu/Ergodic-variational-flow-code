include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")

# tuning
tune = JLD.load("result/eps.jld")
eps = tune["eps"]
Els = tune["ELBOs"]

p_tune = plot(eps, Els, lw = 3, label = "", xlabel = "ϵ", ylabel = "ELBO", title = "Cross", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm) 
savefig(p_tune, "figure/cross_tuning.png")

# ELBO
ELBO = JLD.load("result/elbo_dat.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "Cross", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/cross_elbo.png")

# KSD
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf), labels = false, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Cross", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = false)
savefig(p_ksd, "figure/cross_ksd.png")


# stability plot 
lap = JLD.load("result/stab_lap.jld")
gauss = JLD.load("result/stab_norm.jld")

plot(lap["Ns"], vec(median(lap["fwd_err"], dims=2)), ribbon = get_percentiles(lap["fwd_err"]), lw = 3, label = "Lap Fwd", xlabel = "#refreshments", ylabel = "error", title = "Cross", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, margin=5Plots.mm, legend=:outertopright) 
plot!(lap["Ns"], vec(median(lap["bwd_err"], dims=2)), ribbon = get_percentiles(lap["bwd_err"]), lw = 3, label = "Lap Bwd") 
plot!(lap["Ns"], vec(median(gauss["fwd_err"], dims=2)), ribbon = get_percentiles(gauss["fwd_err"]), lw = 3, label = "Gauss Fwd") 
plot!(lap["Ns"], vec(median(gauss["bwd_err"], dims=2)), ribbon = get_percentiles(gauss["bwd_err"]), lw = 3, label = "Gauss Bwd") 
savefig("figure/cross_stability.png")