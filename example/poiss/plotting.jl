using Distributions, ForwardDiff, LinearAlgebra, Random, Plots
using ErgFlow, JLD, JLD2
include("../../inference/SVI/svi.jl")
include("../common/plotting.jl")
include("../common/result.jl")
include("model.jl")

folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 

colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12]]

###############3
# normal prior
################

# ELBO
ELBO = JLD.load("result/el.jld")
NF_RealNVP3 = JLD2.load("result/RealNVP3_run.jld2")
NF_RealNVP5 = JLD2.load("result/RealNVP5_run.jld2")
NF_RealNVP8 = JLD2.load("result/RealNVP8_run.jld2")
Planar = JLD2.load("result/Planar_run.jld2")
Radial = JLD2.load("result/Radial_run.jld2")

# real nvp 3
println(median(vec(Matrix(NF_RealNVP3["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP3["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP3["elbo"])))))
println(median(vec(Matrix(NF_RealNVP3["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP3["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP3["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP3["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP3["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP3["elbo"])')[2][1])
# real nvp 5
println(median(vec(Matrix(NF_RealNVP5["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP5["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP5["elbo"])))))
println(median(vec(Matrix(NF_RealNVP5["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP5["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP5["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP5["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP5["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP5["elbo"])')[2][1])
# real nvp 8
# all nan
# println(median(vec(Matrix(NF_RealNVP8["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP8["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP8["elbo"])')[1][1])
# println(median(vec(Matrix(NF_RealNVP8["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP8["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP8["elbo"])')[2][1])

# real nvp 3 is best

# planar 5
dat = Matrix(Planar["elbo"][!,"5layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")
# planar 10
dat = Matrix(Planar["elbo"][!,"10layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")
# planar 20
dat = Matrix(Planar["elbo"][!,"20layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")

# planar 5 is best

# radial 5
dat = Matrix(Radial["elbo"][!,"5layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")
# radial 10
dat = Matrix(Radial["elbo"][!,"10layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")
# radial 20
dat = Matrix(Radial["elbo"][!,"20layers"]')
println(median(vec(Matrix(dat))[iszero.(isnan.(vec(Matrix(dat))))]))
println(sum(isnan.(vec(Matrix(dat)))))
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) - get_percentiles(dat)[1][1])
println(median(vec(dat)[iszero.(isnan.(vec(dat)))]) + get_percentiles(dat)[2][1])
println("-----")

# radial 20 is best

# NF = JLD.load("result/nf.jld") 
# el_nf = NF["elbo"]
eps = ELBO["eps"]
Els = Matrix(ELBO["elbos"][:,1:end][2,:]')
Ns = ELBO["Ns"][1:end]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = false, ylabel = "ELBO", xlabel = "#Refreshment",
                        xtickfontsize = 25, ytickfontsize = 25, guidefontsize = 25, legendfontsize = 25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, color = colours[3])
dat = Matrix(NF_RealNVP3["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/poiss_elbo.png")

# full ELBO
Els = ELBO["elbos"][:,1:end]
p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refresh",
                        xtickfontsize = 15, ytickfontsize = 15, guidefontsize = 15, legendfontsize = 15, titlefontsize = 15, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, lincolor = [colours[1] colours[2] colours[3]])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/poiss_elbo_full.png")

# KSD
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"][:,1:end]
Ns = KSD["Ns"][1:end]
nBs = KSD["nBurns"]

NF_KSD = JLD2.load("result/NF_ksd.jld2")
ksd_nf = NF_KSD["ksd"]
labels_nf = ["RealNVP" "Planar" "Radial"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (1., 1000.),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", legend=false, color = colours[3],
            xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, top_margin=5Plots.mm, yaxis=:log)
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS",color = colours[2])
hline!([ksd_nf[3]], linestyle=:dash, lw = 5, label = labels_nf[3],color = colours[6])
hline!([ksd_nf[2]], linestyle=:dash, lw = 5, label = labels_nf[2],color = colours[5])
hline!([ksd_nf[1]], linestyle=:dash, lw = 5, label = labels_nf[1],color = colours[4])
savefig(p_ksd, "figure/poiss_ksd.png")

# p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0.1, 1000.),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Bayesian Poisson Regression", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm, yaxis=:log)
# hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
# savefig(p_ksd, "figure/poiss_ksd_log.png")

NF = JLD2.load("result/RealNVP3.jld2") 
time_trian = NF["train_time"]
Time = JLD.load("result/timing_per_sample.jld")
time_sample_erg_iid = Time["time_sample_erg_iid"]
time_sample_erg_single = Time["time_sample_erg_single"]
time_sample_nf = NF["sampling_time"]
time_sample_hmc = Time["time_sample_hmc"]

NEOtime = JLD2.load("result/neo_time.jld2")["times"] .* 10.
NEOess = JLD2.load("result/ess_neo.jld2")["ess_neo"] ./ 10.
NEOess = NEOess[isnan.(NEOess) .== 0]
colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12], palette(:Greys_3)[2]]

boxplot(["MixFlow iid"], time_sample_erg_iid, label = "MixFlow iid", color = colours[3])
boxplot!(["MixFlow single"], time_sample_erg_single, label = "MixFlow single ", color = :lightblue)
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[4], yscale = :log10, legend = false, guidefontsize=20, tickfontsize=15, xrotation = -10, formatter=:plain, margin=5Plots.mm)
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[1], title = "NF train time= $time_trian (s)")
boxplot!(["NEO"], NEOtime, label = "NEO", color = colours[end])
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)

ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_hmc = ESS["ess_time_hmc"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["MixFlow iid"], ess_time_erg_iid,  label = "MixFlow iid",color = colours[3])
boxplot!(["MixFlow single"], ess_time_erg_single, label = "MixFlow single ", color = :lightblue)
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[1], legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain, margin=5Plots.mm)
boxplot!(["NEO"], NEOess, label = "NEO", color = colours[end])
ylabel!("ESS unit time")

filepath = string("figure/ess.png")
savefig(filepath)

###################
# pairwise plot
###################
NUTS = JLD.load("result/nuts.jld")
NF_nvp = JLD2.load("result/realNVP3.jld2")

D_nuts = NUTS["sample"]
D_nf = NF_nvp["Samples"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]


idx1 = [1:8 ;]
p_vis1 = pairplots(D_nuts[:, idx1], D_ef[1:1000, idx1], D_nf[1:1000, idx1])
savefig(p_vis1, "figure/post_vis1.png")

idx2 = [9:d ;]
p_vis2 = pairplots(D_nuts[:, idx2], D_ef[1:1000, idx2], D_nf[1:1000, idx2])
savefig(p_vis2, "figure/post_vis2.png")