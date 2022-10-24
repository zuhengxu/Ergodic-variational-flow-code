include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/ksd.jl")
include("../common/plotting.jl")
include("../common/result.jl")


folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 

colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12]]

# ELBO
ELBO = JLD.load("result/el_mf.jld")
NF_RealNVP5 = JLD.load("result/RealNVP5_run.jld")
NF_RealNVP10 = JLD.load("result/RealNVP10_run.jld")
Planar = JLD.load("result/Planar_run.jld")
Radial = JLD.load("result/Radial_run.jld")

# real nvp 5
println(median(vec(Matrix(NF_RealNVP5["elbo"]))))
println(sum(isnan.(vec(Matrix(NF_RealNVP5["elbo"])))))
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) - get_percentiles(Matrix(NF_RealNVP5["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) + get_percentiles(Matrix(NF_RealNVP5["elbo"])')[2][1])
# real nvp 10
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP10["elbo"])))))
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP10["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP10["elbo"])')[2][1])

# real nvp 5 is better because it has no nan

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

# radial 10 is best

# el_nf = NF["elbo"]
eps = ELBO["eps"]
Els = ELBO["elbos"]
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = false, ylabel = "ELBO", xlabel = "#Refreshment",
                        xtickfontsize = 25, ytickfontsize = 25, guidefontsize = 25, legendfontsize = 25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, linecolor = [colours[1] colours[3] colours[2]])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"10layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/linreg_elbo.png")

# KSD
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

NF_KSD = JLD.load("result/NF_ksd.jld")
ksd_nf = NF_KSD["ksd"]
labels_nf = ["RealNVP" "Planar" "Radial"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (1., 1000),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", legend=false, color = colours[3],
            xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, top_margin=5Plots.mm, yaxis=:log)
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS",color = colours[2])
hline!([ksd_nf[3]], linestyle=:dash, lw = 5, label = labels_nf[3],color = colours[6])
hline!([ksd_nf[2]], linestyle=:dash, lw = 5, label = labels_nf[2],color = colours[5])
hline!([ksd_nf[1]], linestyle=:dash, lw = 5, label = labels_nf[1],color = colours[4])
savefig(p_ksd, "figure/linreg_ksd.png")

# p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (1., 1000.),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Bayesian Linear Regression", 
#             xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm, yaxis=:log)
# hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
# savefig(p_ksd, "figure/linreg_ksd_log.png")

NF = JLD.load("result/RealNVP5.jld") 
time_trian = NF["train_time"]
Time = JLD.load("result/timing_per_sample.jld")
time_sample_erg_iid = Time["time_sample_erg_iid"]
time_sample_erg_single = Time["time_sample_erg_single"]
time_sample_nf = NF["sampling_time"]
time_sample_hmc = Time["time_sample_hmc"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["ErgFlow iid"], time_sample_erg_iid, label = "ErgFlow iid", color = colours[3])
boxplot!(["ErgFlow single"], time_sample_erg_single, label = "ErgFlow single ", color = :lightblue)
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[4], yscale = :log10, legend = false, guidefontsize=20, tickfontsize=15, xrotation = -10, formatter=:plain, margin=5Plots.mm)
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[1], title = "NF train time= $time_trian (s)")
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)

ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_hmc = ESS["ess_time_hmc"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["ErgFlow iid"], ess_time_erg_iid,  label = "ErgFlow iid",color = colours[3])
boxplot!(["ErgFlow single"], ess_time_erg_single, label = "ErgFlow single ", color = :lightblue)
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[1], legend = false, guidefontsize=20, tickfontsize=15, xrotation = -15, formatter=:plain, margin=5Plots.mm)
ylabel!("ESS unit time")
filepath = string("figure/ess.png")
savefig(filepath)

NUTS = JLD.load("result/nuts.jld")
NF_planar = JLD.load("result/RealNVP5.jld")

D_nuts = NUTS["sample"]
D_nf = NF_planar["Samples"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]

idx1 = [1:5 ;]
p_vis1 = pairplots(D_nuts[:, idx1], D_ef[1:2000, idx1], D_nf[1:2000, idx1])
savefig(p_vis1, "figure/post_vis1.png")

idx2 = [6:d ;]
p_vis2 = pairplots(D_nuts[:, idx2], D_ef[1:2000, idx2], D_nf[1:2000, idx2])
savefig(p_vis2, "figure/post_vis2.png")