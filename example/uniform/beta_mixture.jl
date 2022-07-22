using Plots, Distributions
include("../../inference/util.jl")
#beta mixture
d = MixtureModel([Beta(2.0, 6.0), Beta(6.0, 1.0)], [0.6, 0.4])
f = x-> pdf(d, x)
# Plots.plot(X, f)

shift1(x) = π/16.0
shift2(x) = π/200
shift3(x) = 0.25
shift4(x) = π/200 + 0.25
shift5(n) = rand()

function density_plots(X, T_bar, T_color;labels = ["" "" "" "" "" "" ""], folder::String = "example/uniform/figure/", name::String = "Ergodic.png", kwargs...)
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 
    # N = size(T_bar,1)
    p = Plots.plot(X, ones(length(X)), ylim = (0,3), xlim = (-0.1, 1.1), lw = 2, color = palette(:Reds_4)[4], alpha = 0.7, label = labels[6])
    # Plots.plot!(X, T_bar[:, 5], lw = 2.5, color = T_color[5], alpha = 0.6, label = labels[5], 
    #             guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(X, T_bar[:, 1], lw = 2.5, color = T_color[1], alpha = 0.6, label = labels[1], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    # Plots.plot!(X, T_bar[:, 2], lw = 2.5, color = T_color[2], alpha = 0.8, label = labels[2], 
    #             guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(X, T_bar[:, 3], lw = 2.5, color = T_color[3], alpha = 0.8, label = labels[3], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    # Plots.plot!(X, T_bar[:, 4], lw = 2.5, color = T_color[4], alpha = 0.6, label = labels[4], 
    #             guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(size = (600,300))
    Plots.savefig(p, joinpath(folder, name))
end

function averaged_densities(f, X,N)
    T0 = f.(X)./(N + 1)
    T = [T0 T0 T0 T0 T0]
    for n in 1:N 
        T .+= (1.0/(N + 1)).*shift_densities(f, X, n)
    end
    return T
end


function shifting_trace(x,N)
    T_ir = [(x + n*shift1(x))%1  for n in 0:N]
    T_r = [(x + n*shift3(x))%1  for n in 0:N]
    return T_ir, T_r
end


function sticking_plot(x0, X, N, T_color;labels = ["" "" ""], folder::String = "example/uniform/figure/", name::String = "Ergodic.png", kwargs...)
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 
    T_ir, T_r =shifting_trace(x0, N)
    p = Plots.plot(X, ones(length(X)), ylim = (0,1.5), xlim = (-0.1, 1.1), lw = 4, color = palette(:Reds_4)[4], alpha = 0.8, label = labels[3])
    Plots.plot!(T_ir, 0.99.*ones(N+1), st =:sticks, color = T_color[1], lw =1, alpha = 0.3, label = labels[1],
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(T_r, 0.99.*ones(N+1), st = :sticks, color = T_color[3], lw =1.5, alpha = 1.0,  label = labels[2],
                 guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!([x0], [ones(1)], st =:sticks, markershape = :utriangle, lw = 3, label = "x0 = 0.5", color = :blue)
    Plots.plot!(size = (600,300))
    Plots.savefig(p, joinpath(folder, name))
end



function shift_densities(f, X, N)
    # N = size(T_bar,1)
    Random.seed!(1)
    T1 = @. f((X + N - N*shift1(X) ) % 1)
    T2 = @. f((X + N - N*shift2(X) ) % 1)
    T3 = @. f((X + N - N*shift3(X) ) % 1)
    T4 = @. f((X + N - N*shift4(X) ) % 1)
    r = sum([shift5(n) for n in 1:N])
    T5 = @. f((X + N - r) % 1)
    return [T1 T2 T3 T4 T5]
end

# setting 
# colours = [palette(:Paired_12)[2], palette(:Paired_12)[1], palette(:Paired_12)[9], palette(:Paired_12)[10], palette(:Greens_8)[7]]
colours =[palette(:Greens_7)[7], palette(:Dark2_8)[2], palette(:Dark2_8)[3], palette(:GnBu_7)[7], palette(:Dark2_8)[8]] 
names = ["irrational" "irrational(small)" "rational" "periodic+irrational(small)" "random" "target"]
pdf_names = ["0.6Beta(2,6) + 0.4Beta(6,1)", "0.6Beta(2,6) + 0.4Beta(6,1)", "0.6Beta(2,6) + 0.4Beta(6,1)","0.6Beta(2,6) + 0.4Beta(6,1)","0.6Beta(2,6) + 0.4Beta(6,1)","Unif[0,1]"]
X = [0:0.001:1 ;]

T1_1= f.(X)
T2_1= f.(X)
T3_1= f.(X)
T4_1= f.(X)
T5_1= f.(X)
# bTbar_5 = averaged_densities(f, X, 5)
bTbar_5 = averaged_densities(f, X, 5)
bTbar_20 = averaged_densities(f, X, 20)
# bTbar_50 = averaged_densities(f, X, 50)
bTbar_100 = averaged_densities(f, X, 100)

# names = ["x+π/16 mod 1"  "x+π/400 mod 1" "x+0.25+π/400 mod 1" "x+rand() mod 1"]
density_plots(X, [T1_1 T2_1 T3_1 T4_1 T5_1], colours, name = "beta_average1", title = "Initial density", labels = pdf_names)
density_plots(X, bTbar_5, colours, name = "beta_average5", title = "Averaged density (N = 5)", labels = names)
# density_plots(X, bTbar_10, colours, name = "beta_average10", title = "Averaged density (N = 10)", labels = names)
density_plots(X, bTbar_20, colours, name = "beta_average20", title = "Averaged density (N = 20)", labels = names)
# density_plots(X, bTbar_50, colours, name = "beta_average50", title = "Averaged density (N = 50)", labels = names)
density_plots(X, bTbar_100, colours, name = "beta_average100", title = "Averaged density (N = 100)", labels = names)


T0 = f.(X)
Ts_0 = [T0 T0 T0 T0 T0]
Ts_5 = shift_densities(f, X, 5)
Ts_20 = shift_densities(f, X, 20) 
Ts_100= shift_densities(f, X, 100) 
# Ts_7 = shift_densities(f, X, 7) 

density_plots(X, Ts_0, colours, name = "bt0", title = "Initial density", labels = pdf_names)
density_plots(X, Ts_5, colours, name = "bt5", title = "Transformed density (N = 5)", labels = names )
density_plots(X, Ts_20, colours, name = "bt20", title = "Transformed density (N = 20)")
density_plots(X, Ts_100, colours, name = "bt100", title = "Transformed density (N = 100)")
# density_plots(X, Ts_7, colours, name = "bt7", title = "Transformed density (N = 7)")


# trace of maps
trace_names = ["irrational" "rational" "Unif[0,1]"]
sticking_plot(0.5, X, 0, colours, name = "trace0", title = "Trace of mapped x0 (N = 0)", labels = trace_names)
sticking_plot(0.5, X, 5, colours, name = "trace5", title = "Trace of mapped x0 (N = 5)", labels = trace_names)
sticking_plot(0.5, X, 20, colours, name = "trace20", title = "Trace of mapped x0 (N = 20)", labels = trace_names)
sticking_plot(0.5, X, 100, colours, name = "trace100", title = "Trace of mapped x0 (N = 100)", labels = trace_names)