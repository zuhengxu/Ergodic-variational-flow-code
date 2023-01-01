include("model.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/result.jl")

folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 

colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12]]

# ELBO
ELBO = JLD.load("result/el.jld")
NF_RealNVP5 = JLD2.load("result/RealNVP5_run.jld2")
NF_RealNVP8 = JLD2.load("result/RealNVP8_run.jld2")
Planar = JLD2.load("result/Planar_run.jld2")
Radial = JLD2.load("result/Radial_run.jld2")

# real nvp 5
println(median(vec(Matrix(NF_RealNVP5["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP5["elbo"]))))]))
println(sum(isnan.(vec(Matrix(NF_RealNVP5["elbo"])))))
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) - get_percentiles(Matrix(NF_RealNVP5["elbo"])')[1][1])
println(median(vec(Matrix(NF_RealNVP5["elbo"]))) + get_percentiles(Matrix(NF_RealNVP5["elbo"])')[2][1])
# real nvp 8
# println(median(vec(Matrix(NF_RealNVP8["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP8["elbo"]))))]) - get_percentiles(Matrix(NF_RealNVP8["elbo"])')[1][1])
# println(median(vec(Matrix(NF_RealNVP8["elbo"]))[iszero.(isnan.(vec(Matrix(NF_RealNVP8["elbo"]))))]) + get_percentiles(Matrix(NF_RealNVP8["elbo"])')[2][1])

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

# radial 20 is best

########################3
# ELBO plot
#########################

# el_nf = NF["elbo"]
eps = ELBO["eps"]
Els = Matrix(ELBO["elbos"][:,1:end-1][3,:]')
Ns = ELBO["Ns"][1:end-1]
Labels = Matrix{String}(undef, 1, size(eps, 1))
Labels[1, :].= ["ϵ=$e" for e in eps] 

p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = false, ylabel = "ELBO", xlabel = "#Refreshment",
                        xtickfontsize = 25, ytickfontsize = 25, guidefontsize = 25, legendfontsize = 25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, color = colours[3])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/sp_reg_big_elbo.png")

# full ELBO
Els = ELBO["elbos"][:,1:end-1]
p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refresh",
                        xtickfontsize = 15, ytickfontsize = 15, guidefontsize = 15, legendfontsize = 15, titlefontsize = 15, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, lincolor = [colours[1] colours[2] colours[3]])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
savefig(p_elbo, "figure/sp_reg_big_elbo_full.png")


#####################
# KSD
#####################
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
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (0.8, 1.5),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", legend=false, color = colours[3],
            xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, top_margin=5Plots.mm)
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS", color = colours[2])
hline!([ksd_nf[1]], linestyle=:dash, lw = 5, label = labels_nf[1],color = colours[4]) 
hline!([ksd_nf[2]], linestyle=:dash, lw = 5, label = labels_nf[2],color = colours[5])
hline!([ksd_nf[3]], linestyle=:dash, lw = 5, label = labels_nf[3],color = colours[6])
savefig(p_ksd, "figure/sp_reg_big_ksd.png")

# p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0.1, 1000.),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", title  = "Bayesian Logistic Regression", xtickfontsize=18, ytickfontsize=18, guidefontsize=18, legendfontsize=18, titlefontsize = 18, xrotation = 20, margin=5Plots.mm, yaxis=:log)
# hline!([ksd_nuts], linestyle=:dash, lw = 2, label = "NUTS")
# savefig(p_ksd, "figure/logreg_ksd_log.png")

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

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["EF iid"], time_sample_erg_iid, label = "ErgFlow iid", color = colours[3])
boxplot!(["EF single"], time_sample_erg_single, label = "ErgFlow single ", color = :lightblue)
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[4], yscale = :log10, legend = false, guidefontsize=25, xtickfontsize=25, ytickfontsize=25, titlefontsize=25, xrotation = 20, formatter=:plain, margin=5Plots.mm)
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[1], title = "NF train time= $time_trian (s)")
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)

ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_hmc = ESS["ess_time_hmc"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["EF iid"], ess_time_erg_iid,  label = "ErgFlow iid",color = colours[3])
boxplot!(["EF single"], ess_time_erg_single, label = "ErgFlow single ", color = :lightblue)
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[1], legend = false, guidefontsize=25, xtickfontsize=25, ytickfontsize=25, titlefontsize=25, xrotation = 20, formatter=:plain, margin=5Plots.mm)
ylabel!("ESS unit time")

filepath = string("figure/ess.png")
savefig(filepath)
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

# post pairwise kde +  scatter from NF and EF
for i in 1:Int(ceil(d/10))
    k = i<Int(ceil(d/10)) ? 10*i : d
    idx = [10*(i-1)+1:k ;]
    p_vis = pairplots(D_nuts[:, idx], D_ef[1:2000, idx], D_nf[1:2000, idx])
    savefig(p_vis, "figure/post_vis"*"$(i).png")
end

##################################
# conditional univariate lpdf
###################################
ULPDF = JLD.load("result/univariate_lpdf.jld")
xs = ULPDF["xs"]
lpdf_nf = ULPDF["lpdf_nf"]
lpdf_ef = ULPDF["lpdf_ef"]
lpdf_joint = ULPDF["lpdf_joint"]

lpdf_nf = lpdf_nf .- maximum(lpdf_nf)
lpdf_ef = lpdf_ef .- maximum(lpdf_ef)
lpdf_joint = lpdf_joint .- maximum(lpdf_joint)

ysmall = -2000

plot1 = plot(xs, lpdf_nf, title="NF", lw = 2, ylim = (ysmall, 10), label=false, color = colours[4])
plot2 = plot(xs, lpdf_ef, title="ErgFlow", ylim = (ysmall, 10), lw = 2, label=false, color = colours[3])
plot3 = plot(xs, lpdf_joint, title="Post", ylim = (ysmall, 10), lw = 2, label=false, color = "green")
p = plot(plot1, plot2, plot3, layout = (1, 3), titlefontsize=20, xtickfontsize=20, ytickfontsize=18, xrotation=80)

savefig(p, "figure/univariate_lpdf.png")