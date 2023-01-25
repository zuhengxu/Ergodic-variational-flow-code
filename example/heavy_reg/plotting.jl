include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../../inference/util/ksd.jl")
include("../common/plotting.jl")
include("../common/result.jl")

folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 

colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12], palette(:Set1_6)[6]]

# ELBO
ELBO = JLD.load("result/el_test.jld")
NF_RealNVP5 = JLD2.load("result/RealNVP5_run.jld2")
NF_RealNVP10 = JLD2.load("result/RealNVP10_run.jld2")
Planar = JLD2.load("result/Planar_run.jld2")
Radial = JLD2.load("result/Radial_run.jld2")

# real nvp 5
println(median(vec(Matrix(NF_RealNVP5["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP5["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP5["elbo"])))))
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) - get_percentiles(Matrix(NF_RealNVP5["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) + get_percentiles(Matrix(NF_RealNVP5["elbo"])')[2][1])
# real nvp 10
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP10["elbo"])))))
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP10["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP10["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP10["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP10["elbo"])')[2][1])

# real nvp 5 is best

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

# planar 20 is best

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

# radial 5 is best
########################3
# ELBO plot
#########################

# NF = JLD.load("result/nf.jld") 
# el_nf = NF["elbo"]
eps = ELBO["eps"]
Els = Matrix(ELBO["elbos"][3,:]')
Ns = ELBO["Ns"]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = false, ylabel = "ELBO", xlabel = "#Refresh",
                        xtickfontsize = 25, ytickfontsize = 25, guidefontsize = 25, legendfontsize = 25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, color = colours[3])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/heavy_elbo.png")

# elbo full
Els = ELBO["elbos"]
p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refresh",
                        xtickfontsize = 15, ytickfontsize = 15, guidefontsize = 15, legendfontsize = 15, titlefontsize = 15, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, lincolor = [colours[1] colours[2] colours[3]])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/heavy_elbo_full.png")

#####################
# KSD
#####################
KSD = JLD.load("result/ksd.jld")
ϵ = KSD["ϵ"]
ksd_nuts = KSD["ksd_nuts"]
Ks = KSD["KSD"]
Ns = KSD["Ns"]
nBs = KSD["nBurns"]

NF_KSD = JLD2.load("result/NF_ksd.jld2")
ksd_nf = NF_KSD["ksd"]
labels_nf = ["RealNVP" "Planar" "Radial"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["MixFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (0., Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", legend=false, color = colours[3],
            xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, top_margin=5Plots.mm)#, yaxis=:log)
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS", color = colours[2])
hline!([ksd_nf[1]], linestyle=:dash, lw = 5, label = labels_nf[1],color = colours[4]) 
hline!([ksd_nf[2]], linestyle=:dash, lw = 5, label = labels_nf[2],color = colours[5])
hline!([ksd_nf[3]], linestyle=:dash, lw = 5, label = labels_nf[3],color = colours[6])
savefig(p_ksd, "figure/heavy_ksd.png")

# p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (1., 1000.),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Bayesian Linear Regression", 
#             xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm, yaxis=:log)
# hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
# savefig(p_ksd, "figure/linreg_ksd_log.png")

##########################
# timing plot
###########################
NF = JLD2.load("result/RealNVP5.jld2") 
time_trian = Int(round(NF["train_time"], digits=0))
Time = JLD.load("result/timing_per_sample.jld")
time_sample_erg_iid = Time["time_sample_erg_iid"]
time_sample_erg_single = Time["time_sample_erg_single"]
time_sample_nf = NF["sampling_time"]
time_sample_hmc = Time["time_sample_hmc"]

NEOtime = JLD2.load("result/neo_time.jld2")["times"]
NEOess = JLD2.load("result/neo_time.jld2")["times"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["MixFlow iid"], time_sample_erg_iid, label = "MixFlow iid", color = colours[3])
boxplot!(["MixFlow single"], time_sample_erg_single, label = "MixFlow single ", color = :lightblue)
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[4], yscale = :log10, legend = false, guidefontsize=20, xtickfontsize=20, ytickfontsize=20, titlefontsize=20, xrotation = -20, formatter=:plain, margin=7Plots.mm)
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
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[1], legend = false, guidefontsize=20, xtickfontsize=20, ytickfontsize=20, titlefontsize=20, xrotation = -20, formatter=:plain, margin=7Plots.mm)
boxplot!(["NEO"], NEOess, label = "NEO", color = colours[end])
ylabel!("ESS unit time")

filepath = string("figure/ess.png")
savefig(filepath)

#################################
# conditional univariate lpdf
################################
ULPDF = JLD.load("result/univariate_lpdf.jld")
xs = ULPDF["xs"]
lpdf_nf = ULPDF["lpdf_nf"]
lpdf_ef = ULPDF["lpdf_ef"]
lpdf_joint = ULPDF["lpdf_joint"]

lpdf_nf = lpdf_nf .- maximum(lpdf_nf)
lpdf_ef = lpdf_ef .- maximum(lpdf_ef)
lpdf_joint = lpdf_joint .- maximum(lpdf_joint)

plot1 = plot(xs, lpdf_nf, title="NF", color = colours[4], lw = 2, ylim = (-260, 10), label=false)
plot2 = plot(xs, lpdf_ef, title="MixFlow", color = colours[3], ylim = (-260, 10), lw = 2, label=false)
plot3 = plot(xs, lpdf_joint, title="Post", color = "green", ylim = (-260, 10), lw = 2, label=false)
p = plot(plot1, plot2, plot3, layout = (1, 3), titlefontsize=20, xtickfontsize=20, ytickfontsize=20, xrotation=70)

savefig(p, "figure/univariate_lpdf.png")

###################
# pairwise plot
###################
NUTS = JLD.load("result/nuts_big.jld")
NF_nvp = JLD2.load("result/RealNVP5.jld2")

D_nuts = NUTS["sample"]
D_nf = NF_nvp["Samples"][:, 1:d]
D_ef = JLD.load("result/EF_sample.jld")["sample"]

# psot pairwise kde
for i in 1:Int(ceil(d/10))
    k = i<Int(ceil(d/10)) ? 10*i : d
    idx = [10*(i-1)+1:k ;]
    p_kde = pairkde(D_nuts[:, idx])
    savefig(p_kde, "figure/post_kde"*"$(i).png")
end

# post pairwise kde + scatter from NF and EF
for i in 1:Int(ceil(d/10))
    k = i<Int(ceil(d/10)) ? 10*i : d
    idx = [10*(i-1)+1:k ;]
    p_vis = pairplots(D_nuts[:, idx], D_ef[1:2000, idx], D_nf[1:2000, idx])
    savefig(p_vis, "figure/post_vis"*"$(i).png")
end