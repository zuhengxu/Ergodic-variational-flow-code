include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/metric.jl")
include("../common/plotting.jl")
include("../common/result.jl")

############3
## 2d plots
############3

# tuning
tune = JLD.load("result/eps.jld")
eps = tune["eps"]
Els = tune["ELBOs"]

p_tune = plot(eps, Els, lw = 3, label = "", xlabel = "ϵ", ylabel = "ELBO", title = "Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm) 
savefig(p_tune, "figure/funnel_tuning.png")

# ELBO
ELBO = JLD.load("result/elbo_dat.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/funnel_elbo.png")

# KSD
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = false, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm, legend=(0.5,0.5))
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = false)
savefig(p_ksd, "figure/funnel_ksd.png")


# stability plot
lap = JLD.load("result/stab_lap.jld")
gauss = JLD.load("result/stab_norm.jld")

plot(lap["Ns"], vec(median(lap["fwd_err"], dims=2)), ribbon = get_percentiles(lap["fwd_err"]), lw = 3, label = "Lap Fwd", xlabel = "#refreshments", ylabel = "error", title = "Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, margin=5Plots.mm, legend=:outertopright) 
plot!(lap["Ns"], vec(median(lap["bwd_err"], dims=2)), ribbon = get_percentiles(lap["bwd_err"]), lw = 3, label = "Lap Bwd") 
plot!(lap["Ns"], vec(median(gauss["fwd_err"], dims=2)), ribbon = get_percentiles(gauss["fwd_err"]), lw = 3, label = "Gauss Fwd") 
plot!(lap["Ns"], vec(median(gauss["bwd_err"], dims=2)), ribbon = get_percentiles(gauss["bwd_err"]), lw = 3, label = "Gauss Bwd") 
savefig("figure/funnel_stability.png")







# 5D ELBO
ELBO = JLD.load("result/5d_elbo_dat.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "5D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/5D_funnel_elbo.png")

# 5D KSD
KSD = JLD.load("result/5d_ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "5D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm #=, legend=(0.5,0.5)=#)
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
savefig(p_ksd, "figure/5d_funnel_ksd.png")

# 20D ELBO
ELBO = JLD.load("result/20d_elbo_dat copy.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "20D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/20D_funnel_elbo.png")

# 20D KSD
KSD = JLD.load("result/20d_ksd copy 2.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "20D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm , legend=(0.7,0.3))
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
savefig(p_ksd, "figure/20d_funnel_ksd.png")