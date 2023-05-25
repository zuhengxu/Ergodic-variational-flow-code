include("model.jl")
include("../../inference/NEO/NEO.jl")
include("../../inference/MCMC/NUTS.jl")
include("../common/plotting.jl")
include("../common/result.jl")

folder = "figure"
if ! isdir(folder)
    mkdir(folder)
end 

colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12], palette(:Greys_3)[2], palette(:Set1_8)[8]]

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

#########################
# UHA
#########################
num_rep = 5
mcmcs = [5, 10]
lfrgs = [10, 20, 50]

grid_uha = zeros(Int, size(mcmcs,1) * size(lfrgs,1), 2)

grid_uha[:,1] = vec(repeat(mcmcs, 1, size(lfrgs,1))')
grid_uha[:,2] = repeat(lfrgs, size(mcmcs,1))

uha_ksd = zeros(6, 5)
uha_elbo = zeros(6, 5)

for i in 1:30
    uha_dat = JLD.load("result/uha_sp_reg_big_" * string(i) *".jld")
    if i in [1:5;]
        uha_ksd[1, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[1, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    elseif i in [6:10;]
        uha_ksd[2, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[2, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    elseif i in [11:15;]
        uha_ksd[3, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[3, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    elseif i in [16:20;]
        uha_ksd[4, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[4, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    elseif i in [21:25;]
        uha_ksd[5, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[5, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    elseif i in [26:30;]
        uha_ksd[6, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["ksd"]
        uha_elbo[6, (i % 5) == 0 ? 5 : (i % 5)] = uha_dat["elbo"]
    end
end

# 5, 10
println(grid_uha[1,:])
println(median(uha_elbo[1,:]))
println(median(uha_elbo[1,:]) - get_percentiles(Matrix(uha_elbo[1,:]'))[1][1])
println(median(uha_elbo[1,:]) + get_percentiles(Matrix(uha_elbo[1,:]'))[2][1])

# 5, 20
println(grid_uha[2,:])
println(median(uha_elbo[2,:]))
println(median(uha_elbo[2,:]) - get_percentiles(Matrix(uha_elbo[2,:]'))[1][1])
println(median(uha_elbo[2,:]) + get_percentiles(Matrix(uha_elbo[2,:]'))[2][1])

# 5, 50
println(grid_uha[3,:])
println(median(uha_elbo[3,:]))
println(median(uha_elbo[3,:]) - get_percentiles(Matrix(uha_elbo[3,:]'))[1][1])
println(median(uha_elbo[3,:]) + get_percentiles(Matrix(uha_elbo[3,:]'))[2][1])

# 10, 10
println(grid_uha[4,:])
println(median(uha_elbo[4,:]))
println(median(uha_elbo[4,:]) - get_percentiles(Matrix(uha_elbo[4,:]'))[1][1])
println(median(uha_elbo[4,:]) + get_percentiles(Matrix(uha_elbo[4,:]'))[2][1])

# 10, 20
println(grid_uha[5,:])
println(median(uha_elbo[5,:]))
println(median(uha_elbo[5,:]) - get_percentiles(Matrix(uha_elbo[5,:]'))[1][1])
println(median(uha_elbo[5,:]) + get_percentiles(Matrix(uha_elbo[5,:]'))[2][1])

# 10, 50
println(grid_uha[6,:])
println(median(uha_elbo[6,:]))
println(median(uha_elbo[6,:]) - get_percentiles(Matrix(uha_elbo[6,:]'))[1][1])
println(median(uha_elbo[6,:]) + get_percentiles(Matrix(uha_elbo[6,:]'))[2][1])

# 10, 50 best

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
dat = Matrix(uha_elbo[6,:]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "UHA", ribbon = get_percentiles(dat), color = colours[8])
savefig(p_elbo, "figure/sp_reg_big_elbo.png")

# full ELBO
Els = ELBO["elbos"][:,1:end-1]
p_elbo = plot(reduce(vcat, [[0], Ns]), Els',lw = 5, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refresh",
                        xtickfontsize = 15, ytickfontsize = 15, guidefontsize = 15, legendfontsize = 15, titlefontsize = 15, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, lincolor = [colours[1] colours[2] colours[3]])
dat = Matrix(NF_RealNVP5["elbo"])'
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "RealNVP", ribbon = get_percentiles(dat), color = colours[4])
dat = Matrix(Planar["elbo"][!,"5layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Planar", ribbon = get_percentiles(dat), color = colours[5])
dat = Matrix(Radial["elbo"][!,"20layers"]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "Radial", ribbon = get_percentiles(dat), color = colours[6])
dat = Matrix(uha_elbo[6,:]')
hline!( [median(vec(dat)[iszero.(isnan.(vec(dat)))])], linestyle=:dash, lw = 2, label = "UHA", ribbon = get_percentiles(dat), color = colours[8])
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
labels_nf = ["RealNVP" "Planar" "Radial" "UHA"]

Labels = Matrix{String}(undef, 1, size(nBs, 1))
Labels[1, :].= ["ErgFlow"] 
p_ksd = plot(reduce(vcat, [[0], Ns]), Ks',lw = 5, ylim = (0.8, 1.5),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment", legend=false, color = colours[3],
            xtickfontsize=25, ytickfontsize=25, guidefontsize=25, legendfontsize=25, titlefontsize = 25, xrotation = 20, bottom_margin=10Plots.mm, left_margin=5Plots.mm, top_margin=5Plots.mm)
hline!([ksd_nuts], linestyle=:dash, lw = 5, label = "NUTS", color = colours[2])
hline!([ksd_nf[1]], linestyle=:dash, lw = 5, label = labels_nf[1],color = colours[4]) 
hline!([ksd_nf[2]], linestyle=:dash, lw = 5, label = labels_nf[2],color = colours[5])
hline!([ksd_nf[3]], linestyle=:dash, lw = 5, label = labels_nf[3],color = colours[6])
hline!([median(uha_ksd[6,:])], linestyle=:dash, lw = 5, label = labels_nf[4],color = colours[8])
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
time_sample_nuts = Time["time_sample_nuts"]

# NEOtime = JLD2.load("result/neo_time.jld2")["times"]
NEOess = JLD2.load("result/ess_neo.jld2")

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]
colours = [palette(:Paired_12)[6], palette(:Paired_12)[4], palette(:Paired_12)[2], palette(:Paired_12)[10], palette(:Paired_12)[8], palette(:Paired_12)[12], palette(:Set1_6)[6]]

boxplot(["MixFlow iid"], time_sample_erg_iid, label = "MixFlow iid", color = colours[3])
boxplot!(["MixFlow single"], time_sample_erg_single, label = "MixFlow single", color = :lightblue)
boxplot!(["NF"],time_sample_nf, label = "NF", color = colours[4], yscale = :log10, legend = false, guidefontsize=20, xtickfontsize=20, ytickfontsize=20, titlefontsize=20, xrotation = -20, formatter=:plain, margin=7Plots.mm)
boxplot!(["HMC"], time_sample_hmc, label = "HMC", color = colours[1], title = "NF train time= $time_trian (s)")
boxplot!(["NUTS"], time_sample_nuts, label = "NUTS", color = colours[2])
# boxplot!(["NEO"], NEOtime, label = "NEO", color = colours[end])
ylabel!("time per sample(s)")

filepath = string("figure/sampling_time.png")
savefig(filepath)

ESS = JLD.load("result/ESS.jld")
ess_time_erg_iid = ESS["ess_time_erg_iid"]
ess_time_erg_single = ESS["ess_time_erg_single"]
ess_time_hmc = ESS["ess_time_hmc"]
ess_time_nuts = ESS["ess_time_nuts"]
ess_time_nuts_ad = ESS["ess_time_nuts_ad"]

# colours = [palette(:Paired_8)[5], palette(:Paired_8)[6], palette(:Paired_8)[2], palette(:Paired_8)[1], palette(:Paired_10)[10], palette(:Paired_10)[9], palette(:Paired_8)[4]]

boxplot(["MixFlow iid"], ess_time_erg_iid,  label = "MixFlow iid",color = colours[3])
boxplot!(["MixFlow single"], ess_time_erg_single, label = "MixFlow single ", color = :lightblue)
boxplot!(["HMC"], ess_time_hmc,label = "HMC", color = colours[1], legend = false, guidefontsize=20, xtickfontsize=20, ytickfontsize=20, titlefontsize=20, xrotation = -20, formatter=:plain, margin=7Plots.mm)
# boxplot!(["NEO"], NEOess, label = "NEO", color = colours[end])
boxplot!(["NUTS"], ess_time_nuts[BitVector(abs.(isnan.(ess_time_nuts) .- 1))], label = "NUTS", color = colours[2])
boxplot!(["NUTS_ad"], ess_time_nuts_ad[BitVector(abs.(isnan.(ess_time_nuts_ad) .- 1))], label = "NUTS_ad", color = colours[5])
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
plot2 = plot(xs, lpdf_ef, title="MixFlow", ylim = (ysmall, 10), lw = 2, label=false, color = colours[3])
plot3 = plot(xs, lpdf_joint, title="Post", ylim = (ysmall, 10), lw = 2, label=false, color = "green")
p = plot(plot1, plot2, plot3, layout = (1, 3), titlefontsize=20, xtickfontsize=20, ytickfontsize=18, xrotation=80)

savefig(p, "figure/univariate_lpdf.png")

###################3
# running average plot
#######################
conv = JLD.load("result/running.jld")
m_nuts = conv["m_nuts"]
m_nuts_ad = conv["m_nuts_ad"]
m_hmc = conv["m_hmc"]
m_erg = conv["m_erg"]
m_erg_single = conv["m_erg_single"]
v_nuts = sqrt.(abs.(conv["v_nuts"]))
v_nuts_ad = sqrt.(abs.(conv["v_nuts_ad"]))
v_hmc = sqrt.(conv["v_hmc"])
v_erg = sqrt.(conv["v_erg"])
v_erg_single = sqrt.(conv["v_erg_single"])

n_stop = size(m_nuts, 1)
iters = [1:n_stop ;]

d1 = 1
d2 = 2

p1 = plot(iters, vec(median(m_hmc[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(m_hmc[1:n_stop, d1, :]), lw = 3,label = "HMC", legend = :bottomleft)
    plot!(iters, vec(median(m_erg[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(m_erg[1:n_stop, d1, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(m_erg_single[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(m_erg_single[1:n_stop, d1, :]), lw = 3,label = "MixFlow-S")
    plot!(iters, vec(median(m_nuts[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(m_nuts[1:n_stop, d1, :]),lw = 3, label = "NUTS")
    plot!(iters, vec(median(m_nuts_ad[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(m_nuts_ad[1:n_stop, d1, :]),lw = 3, label = "NUTS adaptive")

p2 = plot(iters, vec(median(m_hmc[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(m_hmc[1:n_stop, d2, :]), lw = 3,label = "HMC", legend = :none)
    plot!(iters, vec(median(m_erg[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(m_erg[1:n_stop, d2, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(m_erg_single[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(m_erg_single[1:n_stop, d2, :]), lw = 3,label = "MixFlow-S")
    plot!(iters, vec(median(m_nuts[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(m_nuts[1:n_stop, d2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(m_nuts_ad[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(m_nuts_ad[1:n_stop, d2, :]), lw = 3,label = "NUTS adaptive")
p = plot(p1, p2, layout = (1, 2), title = "Sparse Regression (high dim)")
savefig(p, "figure/sp_reg_big_mean_est.png")


p1 = plot(iters, vec(median(v_hmc[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(v_hmc[1:n_stop, d1, :]), lw = 3,label = "HMC", legend = :bottomright)
    plot!(iters, vec(median(v_erg[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(v_erg[1:n_stop, d1, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(v_erg_single[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(v_erg_single[1:n_stop, d1, :]), lw = 3,label = "MixFlow-S")
    plot!(iters, vec(median(v_nuts[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(v_nuts[1:n_stop, d1, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_nuts_ad[1:n_stop, d1, :]'; dims =1)), ribbon = get_percentiles(v_nuts_ad[1:n_stop, d1, :]), lw = 3,label = "NUTS adaptive")

p2 = plot(iters, vec(median(v_hmc[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(v_hmc[1:n_stop, d2, :]), lw = 3,label = "HMC", legend = :none)
    plot!(iters, vec(median(v_erg[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(v_erg[1:n_stop, d2, :]), lw = 3,label = "MixFlow", xrotation = 20)
    plot!(iters, vec(median(v_erg_single[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(v_erg_single[1:n_stop, d2, :]), lw = 3,label = "MixFlow-S")
    plot!(iters, vec(median(v_nuts[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(v_nuts[1:n_stop, d2, :]), lw = 3,label = "NUTS")
    plot!(iters, vec(median(v_nuts_ad[1:n_stop, d2, :]'; dims =1)), ribbon = get_percentiles(v_nuts_ad[1:n_stop, d2, :]), lw = 3,label = "NUTS adaptive")
p = plot(p1, p2, layout = (1, 2), title = "Sparse Regression (high dim)")
savefig(p, "figure/sp_reg_big_var_est.png")