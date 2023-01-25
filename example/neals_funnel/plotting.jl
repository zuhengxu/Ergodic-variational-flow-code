include("model_2d.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/result.jl")
using JLD2
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
### neo results---fixed
#id	gamma	Nsteps	stepsize	Nchians	KSD	run_time	NaNratio
# 1	0.2	10	0.5	10	0.055410260886541406	8.631509184	0.0
# 2	0.2	10	0.5	10	0.03609094704972141	9.192355478	0.0
# 3	0.2	10	0.5	10	0.03661090960311319	9.286685976	0.0c
ksd_neo = mean([ 0.055410260886541406, 0.03609094704972141, 0.03661090960311319])
Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["MixFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (0, 0.26),labels = false, ylabel = "Marginal KSD", xlabel = "#Refreshment", xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, margin=5Plots.mm, legend=(0.5,0.5))
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS", legend = false)
hline!( [ksd_neo], linestyle=:dot, lw = 5, label = "NEO", legend=false)
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
savefig(p_elbo, "figure/5d_funnel_elbo.png")

# 5D KSD
KSD = JLD.load("result/5d_ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["MixFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "5D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm #=, legend=(0.5,0.5)=#)
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
savefig(p_ksd, "figure/5d_funnel_ksd.png")

# 20D ELBO
ELBO = JLD.load("result/20d_elbo_dat.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "20D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/20d_funnel_elbo.png")

# 20D KSD
KSD = JLD.load("result/20d_ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["MixFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "20D Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm , legend=(0.7,0.3))
hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
savefig(p_ksd, "figure/20d_funnel_ksd.png")




#######################
# NF results
####################
# modify  figures
ELBO = JLD.load("result/elbo_dat.jld")
NF = JLD.load("result/nf.jld") 

eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
el_nf = NF["elbo"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", 
                title = "Neal's Funnel", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
hline!( [el_nf], linestyle=:dash, lw = 2, label = "NF")
savefig(p_elbo, "figure/funnel_elbo_nf.png")


############3
# saving  scatter plot
##############
x = -30:.1:30
y = -30:.1:30
nf_res = JLD.load("result/nf.jld")
t_nf = nf_res["scatter"]
nf_el = nf_res["elbo"]
pdf_target = (x, y) -> exp(o.logp([x,y]))        
p = contour(x, y, pdf_target, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
scatter!(t_nf[1, :], t_nf[2, :], label = "NF samples", color = 1, legendfontsize= 15, legend=:top) 
savefig(p, "figure/nf_funnel_scatter.png")



############################V
# timing plot
############################
NF = JLD.load("result/NF.jld") 
time_trian = NF["train_time"]
Time = JLD.load("result/timing_per_sample.jld")
time_sample_erg_iid = Time["time_sample_erg_iid"]
time_sample_erg_single = Time["time_sample_erg_single"]
time_sample_nf = Time["time_sample_nf"]
time_sample_nuts = Time["time_sample_nuts"]
time_sample_hmc = Time["time_sample_hmc"]

NEOtime = JLD2.load("result/neo_time.jld2")["times"]
NEOess = JLD2.load("result/neo_time.jld2")["times"]


colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4], palette(:Set1_6)[6]]

boxplot(["MixFlow iid"], time_sample_erg_iid, label = "MixFlow iid", color = colours[1])
boxplot!(["MixFlow single"], time_sample_erg_single, label = "MixFlow single ", color = colours[2])
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[3], yscale = :log10, legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain)
boxplot!(["NUTS"], time_sample_nuts, label = "NUTS", color = colours[4], title = "NF train time= $time_trian (s)")
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[5])
boxplot!(["NEO"], NEOtime, label = "NEO", color = colours[end])
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)


ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_nuts = ESS["ess_time_nuts"]
ess_time_hmc = ESS["ess_time_hmc"]


boxplot(["MixFlow iid"], ess_time_erg_iid,  label = "MixFlow iid",color = colours[1])
boxplot!(["MixFlow single"], ess_time_erg_single, label = "MixFlow single ", color = colours[2])
boxplot!(["NUTS"], ess_time_nuts, label = "NUTS", color = colours[4])
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[5], legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain)
boxplot!(["NEO"], NEOess, label = "NEO", color = colours[end])
ylabel!("ESS unit time")
filepath = string("figure/ess.png")
savefig(filepath)


###################3
# running average plot
#######################
conv = JLD.load("result/running.jld")
m_nuts = conv["m_nuts"]
m_hmc = conv["m_hmc"]
m_erg = conv["m_erg"]
v_nuts = sqrt.(abs.(conv["v_nuts"]))
v_hmc = sqrt.(conv["v_hmc"])
v_erg = sqrt.(conv["v_erg"])

nf_run = JLD.load("result/running_nf.jld")
m_nf = nf_run["m_nf"]
v_nf = sqrt.(nf_run["v_nf"])

iters = [1:size(m_nuts, 1) ;]
p1 = plot(iters, vec(median(m_nuts[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_nuts[:, 1, :]), label = "NUTS", lw = 3)
    plot!(iters, vec(median(m_hmc[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_hmc[:, 1, :]), label = "HMC",lw = 3,  legend = :bottomright)
    plot!(iters, vec(median(m_erg[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_erg[:, 1, :]), label = "MixFlow", lw = 3, xrotation = 20)
    plot!(iters, vec(median(m_nf[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_nf[:, 1, :]), label = "NF", xrotation = 20,lw=3,legendfontsize = 16)
    hline!([0.0],  linestyle=:dash, lw = 2,color =:black,label = "Mean")

p2 = plot(iters, vec(median(m_nuts[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_nuts[:, 2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(m_hmc[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_hmc[:, 2, :]), lw = 3, label = "HMC", legend = :none)
    plot!(iters, vec(median(m_erg[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_erg[:, 2, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(m_nf[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_nf[:, 2, :]), lw = 3,label = "NF", xrotation = 20)
    hline!([0.0],  linestyle=:dash, lw = 2,color = :black, label = "Mean")
p = plot(p1, p2, layout = (1, 2), title = "Neal's Funnel")
savefig(p, "figure/funnel_mean_est.png")


p1 = plot(iters, vec(median(v_nuts[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_nuts[:, 1, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_hmc[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_hmc[:, 1, :]), lw = 3,label = "HMC", legend = :bottomright)
    plot!(iters, vec(median(v_erg[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_erg[:, 1, :]), lw = 3, label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(v_nf[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_nf[:, 1, :]), lw = 3, label = "NF", xrotation = 20, legendfontsize = 16)
    hline!([6],  linestyle=:dash, lw = 2,color = :black, label = "SD")

p2 = plot(iters, vec(median(v_nuts[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_nuts[:, 2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_hmc[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_hmc[:, 2, :]), lw = 3,label = "HMC", legend = :none)
    plot!(iters, vec(median(v_erg[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_erg[:, 2, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(v_nf[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_nf[:, 2, :]), lw = 3,label = "NF", xrotation = 20)
    hline!([10.5],  linestyle=:dash, lw = 2,color = :black, label = "SD")
p = plot(p1, p2, layout = (1, 2), title = "Neal's Funnel")
savefig(p, "figure/funnel_var_est.png")

###########33
# further NF results
############
NF = JLD.load("result/NF_layer.jld")
L = NF["n_layers"]
T_train = NF["T_train"]
T_sample = NF["T_sample"]
E = NF["elbo"]
p1 = plot(L, E, xlabel = "# Layers", ylabel = "ELBO", lw = 3, label = "NF")
hline!([-0.28], linestyle= :dash,  label= "MixFlow", lw =3, legend = :right, legendfontsize = 15)
p2 = plot(L, T_train, xlabel = "# Layers", ylabel = "Training time(s)", lw = 3, legend=:none)
p3 = boxplot(["5"], T_sample[:, 1], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["10"], T_sample[:, 2], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["20"], T_sample[:, 3], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["50"], T_sample[:, 4], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
p = plot(p1, p2, p3, layout= (1, 3))
plot!(size = (1400, 450), xtickfontsize = 15, ytickfontsize = 15,margin=10Plots.mm , guidefontsize= 12)
savefig(p, "figure/nf_layer.png")