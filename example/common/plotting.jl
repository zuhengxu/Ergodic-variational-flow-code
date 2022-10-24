using Plots, Suppressor, JLD, LinearAlgebra, StatsBase, KernelDensity, StatsPlots
using Base:Threads
using ProgressMeter
# using ErgFlowFixed
# include("../../inference/util/metric.jl")
include("../../inference/util/ksd.jl")

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


function scatter_plot(o::ErgodicFlow,x,y;refresh = ErgFlow.pseudo_refresh, contour_plot=false, μ=μ, D= D, ϵ = ϵ, n_sample = 1000, n_mcmc = 500, nB = 250, bins = 500, fig_dir = "figure/", name = "sample.png", show_legend=false) 
    
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
        if show_legend
            pp = plot(p_scatter, legendfontsize=25, xtickfontsize=25, ytickfontsize=25, titlefontsize=25, guidefontsize=25, margin=5Plots.mm, legend=(0.4,0.8))
        else
            pp = plot(p_scatter, legendfontsize=25, xtickfontsize=25, ytickfontsize=25, titlefontsize=25, guidefontsize=25, margin=5Plots.mm, legend=false)
        end
        savefig(pp, joinpath(fig_dir,"contour.png"))
    end    
end


function flow_trace_plot(o::ErgodicFlow,a::HF_params, x,y, xm, ym, z0, ρ0, u0; refresh = pseudo_refresh, inv_ref = inv_refresh, n_mcmc = 500, fig_dir = "figure/", name = "flow_trace.png") 
    
    if ! isdir(fig_dir)
        mkdir(fig_dir)
    end 
    f = (x, y) -> exp(o.logp([x,y]))        
    g = (xm, ym) -> exp(o.lpdf_mom(xm)+ o.lpdf_mom(ym))
    T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd = ErgFlow.flow_trace(o, a, refresh, inv_ref, z0, ρ0, u0, n_mcmc)

    EE = sum(abs2, T_fwd .- T_bwd[end:-1:1, :]; dims = 2) .+ sum(abs2, M_fwd .- M_bwd[end:-1:1, :]; dims = 2) .+ (U_fwd .- U_bwd[end:-1:1]).^2

    p1 = contour(x, y, f, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]), legend=:top)
    plot!(T_fwd[:,1], T_fwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color = :yellow)
    plot!(T_bwd[:,1], T_bwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color= :green, markerstrokewidth=0.6)
    scatter!([T_fwd[1,1]], [T_fwd[1,2]], markershape = :star5, label = "z init", color = :red, markersize = 5)
    scatter!([T_bwd[end,1]], [T_bwd[end,2]], markershape = :star5, label = "z return", color = :orange, markersize = 5)
    p2 = contour(xm, ym, g, colorbar = false, xlim = (xm[1], xm[end]), ylim = (ym[1], ym[end]))
    plot!(M_fwd[:,1], M_fwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color = :yellow)
    plot!(M_bwd[:,1], M_bwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color = :green, markerstrokewidth= 0.6)
    scatter!([M_fwd[1,1]], [M_fwd[1,2]], markershape = :star5, label = "ρ init", color = :red, markersize = 5)
    scatter!([M_bwd[end,1]], [M_bwd[end,2]], markershape = :star5, label = "ρ return", color = :orange, markersize = 5)
    p3 = plot(0.5:0.5:n_mcmc-0.5, sqrt.(EE[end:-1:1]), label = "err", legend = :top, lw = 2)
    scatter!(sqrt.(EE[end:-1:1])[2:2:end], label = "refresh", markersize = 2, alpha = 0.3)
    pp = plot(p1, p2, p3, layout = (1,3) ,plot_title = "ϵ = $(a.leapfrog_stepsize), #lfrg = $(o.n_lfrg) , #Ref = $n_mcmc")
    plot!(size = (1500, 450))
    savefig(pp, joinpath(fig_dir, name))
    return T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd
end

# function flow_trace_plot_rd(o::Union{HF_rd, HF_slice},a::HFrd_params, x,y, xm, ym, z0, ρ0, u0; refresh = ErgFlowFixed.refresh, inv_ref = ErgFlowFixed.inv_refresh, n_mcmc = 500, fig_dir = "figure/", name = "flow_trace.png") 
    
#     if ! isdir(fig_dir)
#         mkdir(fig_dir)
#     end 
#     f = (x, y) -> exp(o.logp([x,y]))        
#     g = (xm, ym) -> exp(o.lpdf_mom(xm)+ o.lpdf_mom(ym))

#     @info "start sampling"
#     T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd = ErgFlowFixed.flow_trace(o, a, refresh, inv_ref, z0, ρ0, u0, n_mcmc)

#     EE = sum(abs2, T_fwd .- T_bwd[end:-1:1, :]; dims = 2) .+ sum(abs2, M_fwd .- M_bwd[end:-1:1, :]; dims = 2) .+ sum(abs2, U_fwd .- U_bwd[end:-1:1, :]; dims  =2)

#     p1 = contour(x, y, f, colorbar = false, xlim = (x[1], x[end]), ylim = (y[1], y[end]), legend=:top)
#     plot!(T_fwd[:,1], T_fwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color = :yellow)
#     plot!(T_bwd[:,1], T_bwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "bwd", color= :green, markerstrokewidth=0.6)
#     scatter!([T_fwd[1,1]], [T_fwd[1,2]], markershape = :star5, label = "z init", color = :red, markersize = 5)
#     scatter!([T_bwd[end,1]], [T_bwd[end,2]], markershape = :star5, label = "z return", color = :orange, markersize = 5)
#     p2 = contour(xm, ym, g, colorbar = false, xlim = (xm[1], xm[end]), ylim = (ym[1], ym[end]))
#     plot!(M_fwd[:,1], M_fwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "fwd", color = :yellow)
#     plot!(M_bwd[:,1], M_bwd[:,2], marker = (:circle, 2.3), alpha = 0.5, label = "bwd", color = :green, markerstrokewidth= 0.6)
#     scatter!([M_fwd[1,1]], [M_fwd[1,2]], markershape = :star5, label = "ρ init", color = :red, markersize = 5)
#     scatter!([M_bwd[end,1]], [M_bwd[end,2]], markershape = :star5, label = "ρ return", color = :orange, markersize = 5)
#     p3 = plot(0.5:0.5:n_mcmc-0.5, sqrt.(EE[end:-1:1]), label = "err", legend = :top, lw = 2)
#     scatter!(sqrt.(EE[end:-1:1])[2:2:end], label = "refresh", markersize = 2, alpha = 0.3)
#     pp = plot(p1, p2, p3, layout = (1,3) ,plot_title = "ϵ = $(a.leapfrog_stepsize), #lfrg = $(o.n_lfrg) , #Ref = $n_mcmc")
#     plot!(size = (1500, 450))
#     savefig(pp, joinpath(fig_dir, name))
#     return T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd
# end



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
        println("elbo = $(Els[i])")
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
    # R = zeros(nsample, nsample)
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
            Ks[i, j+1] = ksd(T_init, o.∇logp)
            # println("n_mcmc = $(Ns[j]) done")
            println("KSD = $(Ks[i, j+1])")
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
    # median_dat = vec(median(dat, dims=1))

    plow = zeros(n)
    phigh = zeros(n)

    for i in 1:n
        dat_remove_nan = (dat[:,i])[iszero.(isnan.(dat[:,i]))]
        median_remove_nan = median(dat_remove_nan)
        plow[i] = median_remove_nan - percentile(vec(dat_remove_nan), p1)
        phigh[i] = percentile(vec(dat_remove_nan), p2) - median_remove_nan
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


function pairplots(D_nuts, T1, T2; bins = 500, c2 = :orange, c1 = :green)
    n = size(D_nuts, 2)
    @assert n ≤ 10
    colnames = ["d"*"$i" for i in 1:n]
    plotter = Matrix{Any}(undef, n,n)
    prog_bar = ProgressMeter.Progress(Int(n*n), dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    # Diagonal with labels
    plotter[1, 1] = histogram(D_nuts[:, 1], ylabel = colnames[1], title = colnames[1], bins = bins, norm = true)
                    density!(D_nuts[:, 1], linewidth = 3)
    ProgressMeter.next!(prog_bar)
    for i in 2 : n
        p_diag = histogram(D_nuts[:, i], bins = bins, norm = true)
        density!(D_nuts[:, i], linewidth = 3)
        plotter[i, i] = p_diag
        ProgressMeter.next!(prog_bar)
    end
    
    # lower diag---bivariate ksd + scatters
    for j in 2 : n
        Xi = T1[:, 1]
        Xj = T1[:, j]
        Yi = T2[:, 1]
        Yj = T2[:, j]
        ylabel = colnames[j]
        k = KernelDensity.kde(D_nuts[:, [1, j]])
        p_scatter = contour(k.x, k.y, k.density', colorbar = false)
        scatter!(Xi, Xj, marker = (:circle, 2.3), alpha = 0.4, color = c1, markerstrokewidth = 0.2)
        scatter!(Yi, Yj, marker = (:circle, 2.3), alpha = 0.3, color = c2, markerstrokewidth= 0.2)
        # scatter!(Xi, Xj, alpha = 0.3, ylabel = ylabel, markercolor =c1, markersize = 1)
        # scatter!(Yi, Yj, alpha = 0.3, markercolor = c2, markersize = 2)

        plotter[1, j] = p_scatter
        plotter[j, 1] = plot(title = colnames[j])
        ProgressMeter.next!(prog_bar)

        # upper diag
        plotter[j, 1]  = contour(k.x, k.y, k.density', colorbar = false)
        ProgressMeter.next!(prog_bar)
    end
            
    for i in 2 : n
        for j in (i + 1) : n
            Xi = T1[:, i]
            Xj = T1[:, j]
            Yi = T2[:, i]
            Yj = T2[:, j]
            k = KernelDensity.kde(D_nuts[:, [i, j]])
            p_scatter = contour(k.x, k.y, k.density', colorbar = false)
            scatter!(Xi, Xj, marker = (:circle, 2.3), alpha = 0.4, color = c1, markerstrokewidth= 0.2)
            scatter!(Yi, Yj, marker = (:circle, 2.3), alpha = 0.3, color = c2, markerstrokewidth= 0.2)
            # scatter!(Xi, Xj, alpha = 0.3, markercolor = c1, markersize = 1)
            # scatter!(Yi, Yj, alpha = 0.3, markercolor = c2, markersize = 1)
            plotter[i, j] = p_scatter
            ProgressMeter.next!(prog_bar)
            # upper diag
            plotter[j, i]  = contour(k.x, k.y, k.density', colorbar = false)
            ProgressMeter.next!(prog_bar)
        end
    end

    p = plot(plotter..., layout=grid(n,n), legend = false)
    plot!(size = (300n, 300n))
    return p
 end


function pairkde(D_nuts; bins = 500)
    n = size(D_nuts, 2)
    colnames = ["d"*"$i" for i in 1:n]
    plotter = Matrix{Any}(undef, n,n)
    prog_bar = ProgressMeter.Progress(Int(n*(n+1)/2), dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)

    # Diagonal with labels
    plotter[1, 1] = histogram(D_nuts[:, 1], ylabel = colnames[1], title = colnames[1], bins = bins, norm = true)
                    density!(D_nuts[:, 1], linewidth = 3)
    ProgressMeter.next!(prog_bar)
    for i in 2 : n
        p_diag = histogram(D_nuts[:, i], bins = bins, norm = true)
        density!(D_nuts[:, i], linewidth = 3)
        plotter[i, i] = p_diag
        ProgressMeter.next!(prog_bar)
    end
    
    # lower diag---bivariate ksd + scatters
    for j in 2 : n
        ylabel = colnames[j]
        k = KernelDensity.kde(D_nuts[:, [1, j]])
        p_scatter = contour(k.x, k.y, k.density', colorbar = false)

        plotter[1, j] = p_scatter
        plotter[j, 1] = plot(title = colnames[j])
        ProgressMeter.next!(prog_bar)
    end
            
    for i in 2 : n
        for j in (i + 1) : n
            k = KernelDensity.kde(D_nuts[:, [i, j]])
            p_scatter = contour(k.x, k.y, k.density', colorbar = false)
            plotter[i, j] = p_scatter
            ProgressMeter.next!(prog_bar)
        end
    end

   # upper diagonal---empty
    for i in 1 : n
        for j in 2 : (i - 1)
            # Xi = T1[:, i]
            # Xj = T1[:, j]
            plotter[i, j] = plot()
            ProgressMeter.next!(prog_bar)
        end
    end


    p = plot(plotter..., layout=grid(n,n), legend = false)
    plot!(size = (300n, 300n))
    return p
 end
