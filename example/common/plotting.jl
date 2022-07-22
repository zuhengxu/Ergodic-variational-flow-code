using Plots, Suppressor, JLD, LinearAlgebra, StatsBase
include("../../inference/util/metric.jl")

function vis_hist(X, logp, samples; bins = 200, kwargs...)
    curve = exp.([logp(x) for x in X])
    p = Plots.histogram(samples, normed = :pdf, bins = bins, alpha = 0.5, label = "")
    Plots.plot!(X, curve, xlim = (X[1], X[end]),guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
end

# ELBO plot for ErgFlow
function ELBO_plot(o::ErgFlow.HamFlow, o1::SVI.MFGauss; μ = μ, D = D, eps, Ns, nBs, elbo_size = 1000, 
                    fig_dir = "figure/", fig_name = "elbo.png", res_dir = "result/", res_name = "elbo_dat.jld", 
                    kwargs...)
    # # generate folder 
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    Els = zeros(size(eps, 1), size(Ns, 1)+1)
    Els[:, 1] .= SVI.ELBO(o1, μ, D; elbo_size = elbo_size)*ones(size(eps, 1))
    # Els[:, 1] .= SVI.ELBO(o1, zeros(d), ones(d); elbo_size = 1000) *ones(size(eps, 1))
    for i in 1:size(eps, 1)
        for j in 1:size(Ns, 1)
            nB = nBs[j]
            println("nBurn = $nB")
            Els[i, j+1] = ErgFlow.ELBO(o,eps[i]*ones(o.d), μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord,Ns[j]; 
                                nBurn = nB, elbo_size = elbo_size, print = true)
            println("ϵ = $(eps[i]), n_mcmc = $(Ns[j]) done")
            # Els[i, j+1] = ErgFlow.ELBO(o, eps[i]*ones(d), zeros(d), ones(d), ErgFlow.pseudo_refresh_coord, ErgFlow.inv_refresh_coord,Ns[j]; nBurn = Int(ceil(Ns[j]/2)), elbo_size = 500)
        end
    end 
    JLD.save(joinpath(res_dir, res_name), "elbos", Els, "eps", eps, "Ns", Ns, "nBurns", nBs)
    Labels = Matrix{String}(undef, 1, size(eps, 1))
    Labels[1, :] .= ["ϵ=$e" for e in eps] 
    p = plot(reduce(vcat, [[0], Ns]), Els',lw = 3, labels = Labels, legend = :outertopright, ylabel = "ELBO", xlabel = "#Refreshment"; kwargs...)
    savefig(p, joinpath(fig_dir, fig_name))
    # savefig(p, "example/banana/figure/els_$n_lfrg.png")
end


function Burn_plot(o::ErgFlow.HamFlow; μ = μ, D = D, ϵ, n_mcmc, nBs, elbo_size = 1000, 
                    fig_dir = "figure/", fig_name = "burn.png", res_dir = "result/", res_name = "burn.jld", 
                    kwargs...) 

    # # generate folder 
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    Els = zeros(size(nBs, 1))
    # Els[:, 1] .= SVI.ELBO(o1, zeros(d), ones(d); elbo_size = 1000) *ones(size(eps, 1))
    @threads for i in 1:size(nBs, 1)
        nB = nBs[i]
        Els[i] = ErgFlow.ELBO(o, ϵ, μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord,n_mcmc; 
                                nBurn = nB, elbo_size = elbo_size, print = true)
    end 
    p = plot(nBs, Els,lw = 3, title = "ϵ = $ϵ, lfrg = $(o.n_lfrg), n_mcmc = $n_mcmc", legend = :outertopright, ylabel = "ELBO", xlabel = "#Burn"; kwargs...)
    # savefig(p, "example/banana/figure/els_$n_lfrg.png")
    JLD.save(joinpath(res_dir, res_name), "elbos", Els, "ϵ", ϵ, "n_mcmc", n_mcmc, "nBurns", nBs)
    savefig(p, joinpath(fig_dir, fig_name))
end


function scatter(o::ErgodicFlow,x,y;refresh = ErgFlow.pseudo_refresh, contour_plot=false, μ=μ, D= D, ϵ = ϵ, n_sample = 1000, n_mcmc = 500, nB = 250, bins = 500, fig_dir = "figure/", name = "sample.png") 
    
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    # if ! isdir(res_dir)
    #     mkdir(res_dir)
    # end 
    a = HF_params(ϵ, μ, D) # using learned VI parameters
    # x = -30:.1:30
    # y = -15:.1:60
    f = (x, y) -> exp(o.logp([x,y]))        
    gsvi = (x, y) -> exp(o.logq0([x, y], μ, D))
    
    # T, M, U = ErgFlow.Sampler(o,a,ErgFlow.pseudo_refresh_coord,n_mcmc,n_sample; nBurn = nB)
    T, M, U = ErgFlow.Sampler(o,a, refresh, n_mcmc,n_sample; nBurn = nB)
    p_scatter = contour(x, y, f, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
    scatter!(T[:,1], T[:,2], alpha = 0.3, label = "ErgFlow")
    # p2 = contour(-10:0.1:10, -10:0.1:10, f_m, lw = 3, colorbar =false, levels = 30)
    # p2 = plot(-8:0.1:8, o.pdf_mom, lw = 4, label = "Laplace")
    p2 = plot(-8:0.1:8, o.pdf_mom, lw = 4, label = "normal")
    # p2 = plot(-8:0.1:8, ErgFlow.pdf_logistic_std, lw = 4, label = "Logistic")
    # scatter!(M[1,:], M[2,:], alpha = 0.3,c=:green, label = "Momumtum")
    histogram!(M[1,:], alpha = 0.25, bins = bins, normed = true, label = "Momumtum (1)")
    histogram!(M[2,:], alpha = 0.25, bins = bins, normed = true, label = "Momumtum (2)")
    p3 = plot(0:0.1:1, one.([0:0.1:1 ;]), lw = 4, label = "Unif[0,1]")
    histogram!(U, alpha = 0.25, bins = 100, normed = true, label = "Pseudotime")
    pp = plot(p_scatter, p2, p3, layout = (1,3) ,plot_title = "ϵ = $ϵ, #lfrg = $(o.n_lfrg) , #Ref = $n_mcmc, #Burn = $nB")
    plot!(size = (1500, 450))
    savefig(pp, joinpath(fig_dir, name))
    
    if contour_plot 
        p2 = contour(x, y, gsvi, colorbar = false, title = "MF Gaussian fit (q₀)",xlim = (x[1], x[end]), ylim = (y[1], y[end]))
        pp = plot(p2, p_scatter, layout = (1,2))
        plot!(size = (600, 200))
        savefig(pp, joinpath(fig_dir,"contour.png"))
    end    
end


function ef_eps(X, Y, o;μ, D, n_mcmc, elbo_size=1000) 
    n1, n2 = size(X, 1), size(Y, 1)
    Els = Matrix{Float64}(undef, n1, n2)
    for i in 1:n1
        for j in 1:n2
            Els[i, j] = ErgFlow.ELBO(o, [X[i], Y[j]], μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord,n_mcmc; 
                        nBurn = 0, elbo_size = elbo_size, print = true)
        end
    end 
    return Els
end

function eps_tunning(eps,o; μ, D, n_mcmc, elbo_size=1000, nB  =0, 
                    fig_dir = "figure/", fig_name = "eps_tune.png", res_dir = "result/", res_name = "eps.jld", 
                    kwargs...)
    # # generate folder 
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    n = size(eps,1)
    Els = Vector{Float64}(undef, n)
    for i in 1:n 
        println("$i/$n")
        Els[i] = ErgFlow.ELBO(o, eps[i]*ones(d), μ, D, ErgFlow.pseudo_refresh_coord,ErgFlow.inv_refresh_coord, n_mcmc; 
                        nBurn = nB, elbo_size = elbo_size, print = true)
    end
    p = plot(eps, Els, lw = 3, label = "", xlabel = "ϵ", ylabel = "ELBO"; kwargs...) 
    plot!(size = (800, 450))
    JLD.save(joinpath(res_dir, res_name), "ELBOs",Els, "eps", eps, "nBurn", nB)
    savefig(p, joinpath(fig_dir, fig_name))
    return Els
end

function ksd_plot(o::ErgFlow.HamFlow; μ::Vector{Float64}, D::Vector{Float64}, ϵ::Vector{Float64}, Ns, nBs,  
                    ref::Function = ErgFlow.pseudo_refresh_coord, nsample = 2000, 
                    fig_dir = "figure/", fig_name = "ksd.png", res_dir = "result/", res_name = "ksd.jld", 
                    kwargs...)

    # # generate folder 
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    D_nuts = nuts(μ, 0.7, logp, ∇logp, 5000, 10000)
    R = zeros(nsample, nsample)
    ksd_nuts = ksd(D_nuts, ∇logp) 
    println("KSD_nuts = $ksd_nuts")

    Ks = zeros(size(nBs, 1), size(Ns, 1)+1)
    T_init =  μ'.+ D' .* randn(nsample, d) 
    Ks[:, 1] .= ksd(T_init, o.∇logp) 
    # Els[:, 1] .= SVI.ELBO(o1, zeros(d), ones(d); elbo_size = 1000) *ones(size(eps, 1))
    for i in 1:size(nBs, 1)
        for j in 1:size(Ns, 1)
            nB = nBs[i]
            n_mcmc = Ns[j]
            println("nBurn = $nB, nRef = $n_mcmc")
            a = ErgFlow.HF_params(ϵ, μ, D) 
            # taking samples for fixed setting
            ErgFlow.Sampler!(T_init, o, a, ref, n_mcmc, nsample; nBurn = nB)
            Ks[i, j+1] = ksd!(R, T_init, o.∇logp)
            println("n_mcmc = $(Ns[j]) done")
        end
    end 
    Labels = Matrix{String}(undef, 1, size(nBs, 1))
    # Labels[1, :].= ["ErgFlow(#Burn=$b)" for b in nBs] 
    Labels[1, :].= ["ErgFlow"] 
    p = plot(reduce(vcat, [[0], Ns]), Ks',lw = 3, ylim = (0, Inf),labels = Labels, ylabel = "Marginal KSD", xlabel = "#Refreshment"; kwargs...)
    hline!([ksd_nuts], linestype=:dash, lw = 2, label = "NUTS")
    # savefig(p, "example/banana/figure/els_$n_lfrg.png")
    JLD.save(joinpath(res_dir, res_name), "KSD", Ks, "ksd_nuts",ksd_nuts,  "ϵ",ϵ, "Ns", Ns, "nBurns", nBs)
    savefig(p, joinpath(fig_dir, fig_name))
    return Ks
end


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



# ################3
# # gif for vis 
# ################
function scatter_gif(o::ErgodicFlow, a::HF_params, x,y; 
                    momentum::String = "Normal",refresh::Function = pseudo_refresh, 
                    n_sample = 100, n_mcmc = 500, bins = 50, freq = 10, 
                    fig_dir = "figure/", name = "sample.gif") 

    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    f = (x, y) -> exp(o.logp([x,y]))        
    
    T_collect, M_collect = [], []
    for i in 1:n_sample    
        z0 = randn(o.d) .* a.D .+ a.μ
        T_lfrg, M_lfrg, _, _, _ = ErgFlow.flow_fwd_save(o,a.leapfrog_stepsize, refresh, z0, o.ρ_sampler(o.d), rand(), n_mcmc; freq = freq)
        push!(T_collect, T_lfrg)
        push!(M_collect, M_lfrg)
        print(i,"/",n_sample, "\r")
        flush(stdout)
    end
    TT = reduce((x, y)->cat(x, y; dims = 3), T_collect)
    MM = reduce((x, y)->cat(x, y; dims = 3), M_collect)

    @info "generating gif"
    # gif 
    anim = @suppress_err @animate for i=1:size(TT, 1)
        print(i,"/", size(TT, 1), "\r")
        flush(stdout)
        p1 = contour(x, y, f, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]))
        scatter!(TT[i,1,:], TT[i,2,:], alpha = 1.0, label = "ErgFlow")
        # p2 = contour(-10:0.1:10, -10:0.1:10, f_m, lw = 3, colorbar =false, levels = 30)
        # p2 = plot(-8:0.1:8, o.pdf_mom, lw = 4, label = "Laplace")
        p2 = plot(-8:0.1:8, o.pdf_mom, lw = 4, label = momentum)
        # p2 = plot(-8:0.1:8, ErgFlow.pdf_logistic_std, lw = 4, label = "Logistic")
        # scatter!(M[1,:], M[2,:], alpha = 0.3,c=:green, label = "Momumtum")
        histogram!(MM[i, 1,:], alpha = 0.25, bins = bins, normed = true, label = "Mom (1)")
        histogram!(MM[i, 2,:], alpha = 0.25, bins = bins, normed = true, label = "Mom (2)")
        # p3 = plot(0:0.1:1, one.([0:0.1:1 ;]), lw = 4, label = "Unif[0,1]")
        # histogram!(U, alpha = 0.25, bins = 100, normed = true, label = "Pseudotime")
        pp = plot(p1, p2, layout = (1,2),plot_title = " #lfrg = $(i*freq)/$(o.n_lfrg*n_mcmc) , #Ref = $n_mcmc")
        plot!(size = (1000, 450))
    end
    gif(anim, joinpath(fig_dir,name), fps = 20)
end

