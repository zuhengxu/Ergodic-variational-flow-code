using StatsPlots, JLD
include("model_2d.jl")
# include("../../inference/util/metric.jl")
include("../common/plotting.jl")

# tuning
tune = JLD.load("result/eps.jld")
eps = tune["eps"]
Els = tune["ELBOs"]

p_tune = plot(eps, Els, lw = 3, label = "", xlabel = "ϵ", ylabel = "ELBO", title = "Banana", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm) 
savefig(p_tune, "figure/banana_tuning.png")

# ELBO
ELBO = JLD.load("result/elbo_dat.jld")
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "Banana", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
savefig(p_elbo, "figure/banana_elbo.png")

# KSD
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, margin=5Plots.mm)
hline!( [ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS", legend=false)
savefig(p_ksd, "figure/banana_ksd.png")

lap = JLD.load("result/stab_lap.jld")
gauss = JLD.load("result/stab_norm.jld")
logis =  JLD.load("result/stab_log.jld")
# gauss_new = JLD.load("result/stab_norm_new.jld")

plot(lap["Ns"], vec(median(lap["fwd_err"], dims=2)), ribbon = get_percentiles(lap["fwd_err"]), lw = 3, label = "Lap Fwd", xlabel = "#refreshments", ylabel = "error", title = "Banana", 
    xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, legend=:outertopright) 
plot!(lap["Ns"], vec(median(lap["bwd_err"], dims=2)), ribbon = get_percentiles(lap["bwd_err"]), lw = 3, label = "Lap Bwd") 
plot!(lap["Ns"], vec(median(logis["fwd_err"], dims=2)), ribbon = get_percentiles(logis["fwd_err"]), lw = 3, label = "Logis Fwd", xlabel = "#refreshments", ylabel = "error", title = "Banana", 
    xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 45, legend = :outertopright) 
plot!(lap["Ns"], vec(median(logis["bwd_err"], dims=2)), ribbon = get_percentiles(logis["bwd_err"]), lw = 3, label = "Logis Bwd") 
plot!(lap["Ns"], vec(median(gauss["fwd_err"], dims=2)), ribbon = get_percentiles(gauss["fwd_err"]), lw = 3, label = "Gauss Fwd") 
plot!(lap["Ns"], vec(median(gauss["bwd_err"], dims=2)), ribbon = get_percentiles(gauss["bwd_err"]), lw = 3, label = "Gauss Bwd") 
# plot!(lap["Ns"], vec(median(gauss_new["fwd_err"], dims=2)), ribbon = get_percentiles(gauss_new["fwd_err"]), lw = 3, label = "Gauss1 Fwd") 
# plot!(lap["Ns"], vec(median(gauss_new["bwd_err"], dims=2)), ribbon = get_percentiles(gauss_new["bwd_err"]), lw = 3, label = "Gauss1 Bwd") 
savefig("figure/banana_stability.png")



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

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment", title = "Banana", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm)
hline!( [el_nf], linestyle=:dash, lw = 2, label = "NF")
savefig(p_elbo, "figure/banana_elbo_nf.png")

# ############3
# # saving  scatter plot
# ##############

x = -20:.1:20
y = -15:.1:30
nf_res = JLD.load("result/nf.jld")
t_nf = nf_res["scatter"]
nf_el = nf_res["elbo"]
pdf_target = (x, y) -> exp(o.logp([x,y]))        
p = contour(x, y, pdf_target, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
scatter!(t_nf[1, :], t_nf[2, :], label = "NF samples", color = 1, legendfontsize= 15, legend=:top)
savefig(p, "figure/nf_cross_scatter.png")



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

colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["ErgFlow iid"], time_sample_erg_iid, label = "ErgFlow iid", color = colours[1])
boxplot!(["ErgFlow single"], time_sample_erg_single, label = "ErgFlow single ", color = colours[2])
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[3], yscale = :log10, legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain)
boxplot!(["NUTS"], time_sample_nuts, label = "NUTS", color = colours[4], title = "NF train time= $time_trian (s)")
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[5])
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)


ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_nuts = ESS["ess_time_nuts"]
ess_time_hmc = ESS["ess_time_hmc"]

colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["ErgFlow iid"], ess_time_erg_iid,  label = "ErgFlow iid",color = colours[1])
boxplot!(["ErgFlow single"], ess_time_erg_single, label = "ErgFlow single ", color = colours[2])
boxplot!(["NUTS"], ess_time_nuts, label = "NUTS", color = colours[4])
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[5], legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain)
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
p1 = plot(iters, vec(median(m_nuts[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_nuts[:, 1, :]),lw = 3, label = "NUTS")
    plot!(iters, vec(median(m_hmc[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_hmc[:, 1, :]), lw = 3,label = "HMC", legend = :bottomleft)
    plot!(iters, vec(median(m_erg[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_erg[:, 1, :]), lw = 3,label = "ErgFlow", xrotation = 20)
    plot!(iters, vec(median(m_nf[:, 1, :]'; dims =1)), ribbon = get_percentiles(m_nf[:, 1, :]), lw = 3,label = "NF", xrotation = 20, legendfontsize = 13)
    hline!([0.0],  linestyle=:dash, lw = 2,color =:black,label = "Mean")

p2 = plot(iters, vec(median(m_nuts[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_nuts[:, 2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(m_hmc[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_hmc[:, 2, :]), lw = 3,label = "HMC", legend = :none)
    plot!(iters, vec(median(m_erg[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_erg[:, 2, :]), lw = 3,label = "ErgFlow", xrotation = 20)
    plot!(iters, vec(median(m_nf[:, 2, :]'; dims =1)), ribbon = get_percentiles(m_nf[:, 2, :]), lw = 3,label = "NF", xrotation = 20)
    hline!([0.0],  linestyle=:dash, lw = 2,color = :black, label = "Mean")
p = plot(p1, p2, layout = (1, 2), title = "Banana")
savefig(p, "figure/banana_mean_est.png")


p1 = plot(iters, vec(median(v_nuts[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_nuts[:, 1, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_hmc[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_hmc[:, 1, :]), lw = 3,label = "HMC", legend = :bottomright)
    plot!(iters, vec(median(v_erg[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_erg[:, 1, :]), lw = 3,label = "ErgFlow", xrotation = 20)
    plot!(iters, vec(median(v_nf[:, 1, :]'; dims =1)), ribbon = get_percentiles(v_nf[:, 1, :]), lw = 3,label = "NF", xrotation = 20, legendfontsize = 16)
    hline!([10],  linestyle=:dash, lw = 2,color = :black, label = "SD")

p2 = plot(iters, vec(median(v_nuts[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_nuts[:, 2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_hmc[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_hmc[:, 2, :]), lw = 3,label = "HMC", legend = :none)
    plot!(iters, vec(median(v_erg[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_erg[:, 2, :]), lw = 3,label = "ErgFlow", xrotation = 20)
    plot!(iters, vec(median(v_nf[:, 2, :]'; dims =1)), ribbon = get_percentiles(v_nf[:, 2, :]),lw = 3, label = "NF", xrotation = 20)
    hline!([sqrt(186)],  linestyle=:dash, lw = 2,color = :black, label = "SD")
p = plot(p1, p2, layout = (1, 2), title = "Banana")
savefig(p, "figure/banana_var_est.png")

###########33
# further NF results
############
NF = JLD.load("result/NF_layer.jld")
L = NF["n_layers"]
T_train = NF["T_train"]
T_sample = NF["T_sample"]
E = NF["elbo"]
p1 = plot(L, E, xlabel = "# Layers", ylabel = "ELBO", lw = 3, label = "NF")
hline!([-0.36], linestyle= :dash,  label= "ErgFlow", lw =3, legend = :topright, legendfontsize = 15)
p2 = plot(L, T_train, xlabel = "# Layers", ylabel = "Training time(s)", lw = 3, legend=:none)
p3 = boxplot(["5"], T_sample[:, 1], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["10"], T_sample[:, 2], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["20"], T_sample[:, 3], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
boxplot!(["50"], T_sample[:, 4], xlabel = "# Layers", ylabel = "Per sample time(s)", yscale=:log10, legend = :none)
p = plot(p1, p2, p3, layout= (1, 3))
plot!(size = (1400, 450), xtickfontsize = 15, ytickfontsize = 15,margin=10Plots.mm , guidefontsize= 12)
savefig(p, "figure/nf_layer.png")