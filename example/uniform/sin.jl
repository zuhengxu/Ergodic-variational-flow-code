using Plots
include("../../inference/util.jl")


shift1(x) = π/16.0
shift2(x) = π/16.0*0.04
shift3(x) = π/16.0*0.04 + 0.05
shift4(n) = rand()
# shift3(x) = sin(2.0*x) 

function init(x)
    return sin(2*x*π)*0.5 + 1
end

function shift_density(x::Float64, shift::Function)
    return init(x - shift(x))
end


function averaged_density(X, shift, N)
    T = init.(X)./N
    for n in 1:N 
        T .+= @. 1.0/(N + 1)*init(X - n*shift(X))
    end
    return T
end


function averaged_density_rand(X, shift, N)
    Random.seed!(1)
    T = init.(X)./N
    for n in 1:N 
        r = shift(n)
        T .+= @. 1.0/(N + 1)*init(X - r)
    end
    return T
end

function density_plots(X, T_bar, T_color;labels = ["" "" "" "" ""], folder::String = "figure/", name::String = "Ergodic.png", kwargs...)
    # create the figure folder
    if ! isdir(folder)
        mkdir(folder)
    end 
    # N = size(T_bar,1)
    p = Plots.plot(X, ones(length(X)), ylim = (0.3, 1.6), xlim = (0, 1), lw = 3, color = palette(:Reds_4)[4], alpha = 0.8, label = labels[5])
    Plots.plot!(X, T_bar[:, 1], lw = 3, color = T_color[1], alpha = 0.7, label = labels[1], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(X, T_bar[:, 2], lw = 3, color = T_color[2], alpha = 0.7, label = labels[2], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(X, T_bar[:, 3], lw = 3, color = T_color[3], alpha = 0.7, label = labels[3], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(X, T_bar[:, 4], lw = 3, color = T_color[4], alpha = 0.7, label = labels[4], 
                guidefontsize=20, ytickfontsize=15, xtickfontsize=15; kwargs...)
    Plots.plot!(size = (600,300))
    Plots.savefig(p, joinpath(folder, name))
end


function shift_densities(X, N)
    # N = size(T_bar,1)
    Random.seed!(1)
    T1 = @. init(X - N*shift1(X))
    T2 = @. init(X - N*shift2(X))
    T3 = @. init(X - N*shift3(X))
    r = shift4(N)
    T4 = @. init(X - r)
    return [T1 T2 T3 T4]
end

# setting 
colours = [palette(:Paired_8)[2], palette(:Paired_10)[10], palette(:Paired_8)[4], palette(:darkrainbow)[4]]
names = ["irrational" "irrational(small)" "periodic+irrational(small)" "random" "target"]
X = [0:0.001:1 ;]

T1_1= init.(X)
T1_10 = averaged_density(X, shift1, 10)
T1_20 = averaged_density(X, shift1, 20)
T1_100  = averaged_density(X, shift1, 100)
T2_1= init.(X)
T2_10 = averaged_density(X, shift2, 10)
T2_20 = averaged_density(X, shift2, 20)
T2_100  = averaged_density(X, shift2, 100)
T3_1= init.(X)
T3_10 = averaged_density(X, shift3, 10)
T3_20 = averaged_density(X, shift3, 20)
T3_100  = averaged_density(X, shift3, 100)
T4_1= init.(X)
T4_10 = averaged_density_rand(X, shift4, 10)
T4_20 = averaged_density_rand(X, shift4, 20)
T4_100  = averaged_density_rand(X, shift4, 100)

pdf_names = ["sin(2πx)/2+1", "sin(2πx)/2+1","sin(2πx)/2+1","sin(2πx)/2+1","Unif[0,1]"]
# names = ["x+π/16 mod 1"  "x+π/400 mod 1" "x+0.25+π/400 mod 1" "x+rand() mod 1"]
density_plots(X, [T1_1 T2_1 T3_1 T4_1], colours, name = "average1", title = "Initial density", labels = pdf_names)
density_plots(X, [T1_10 T2_10 T3_10 T4_10], colours, name = "average10", title = "Averaged density (N = 10)", labels = names)
density_plots(X, [T1_20 T2_20 T3_20 T4_20], colours, name = "average20", title = "Averaged density (N = 20)")
density_plots(X, [T1_100 T2_100 T3_100 T4_100], colours, name = "average100", title = "Averaged density (N = 100)")



T0 = init.(X)
Ts_1 = [T0 T0 T0 T0]
Ts_10 = shift_densities(X, 10)
Ts_20 = shift_densities(X, 20) 
Ts_100 = shift_densities(X, 100) 

density_plots(X, Ts_1, colours, name = "t1", title = "Initial density", labels = pdf_names)
density_plots(X, Ts_10, colours, name = "t10", title = "Transformed density (N = 10)", labels = names )
density_plots(X, Ts_20, colours, name = "t20", title = "Transformed density (N = 20)")
density_plots(X, Ts_100, colours, name = "t100", title = "Transformed density (N = 100)")