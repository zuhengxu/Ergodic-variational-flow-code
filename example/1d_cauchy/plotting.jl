include("main.jl")
###################
# Sample histogram
##################
# hist gram
function hist(T,folder,name, kwargs...)
    p = vis_hist([-6: 0.1:9 ;], logp, T; bins = 1000, label = "Cauchy(0,1)", lw = 3, kwargs...)
    Plots.plot!(size = (600,300), title = "Sample Histogram (N = 10000)")
    Plots.savefig(p, joinpath(folder, name))
end
# forward sample
# z1, ρ1, u1= ErgFlow.flow_fwd(Flow, λ, z0, ρ0, u0, n_mcmc)
# ErgFlow.single_error_checking(Flow, λ, z1, ρ1, u1, n_mcmc)
# z1, ρ1, u1, T1 = ErgFlow.flow_fwd_track(Flow, λ, z0, ρ0, u0, n_mcmc)
# hist(T1[:, 1])

# sample histogram 
T_sample = ErgFlow.Sampler(Flow, λ, n_mcmc, N)
hist(T_sample[:, 1], "figure/", "sample_hist.png")



###################
# Sample histogram
##################
import PlotlyJS
# prepare interactive 3D plots on joint target
log_joint_den = (x,y) -> logp(x) + ErgFlow.lpdf_laplace_std(y)
xs = [-5.01:0.1:5 ;]
ys = [-4.01:0.1:4 ;]
z_dat = zeros(size(xs,1), size(ys,1))
for i in 1:size(xs,1)
    for j in 1:size(ys,1)
        z_dat[i,j] = log_joint_den(xs[i], ys[j])
    end
end
layout = PlotlyJS.Layout(
    width=500, height=500,
    scene = PlotlyJS.attr(
    xaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    yaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    zaxis = PlotlyJS.attr(showticklabels=false, visible=false),
    ),
    margin=PlotlyJS.attr(l=0, r=0, b=0, t=0, pad=0),
    colorscale = "Vird"
)
# true log joint target
p_target = PlotlyJS.plot(PlotlyJS.surface(z=z_dat, x=xs, y=ys, showscale=false), layout)
PlotlyJS.savefig(p_target, joinpath(folder,"target_lpdf.png"))

p_ave_pdf = PlotlyJS.plot(PlotlyJS.surface(z=Ds, x=xs, y=ys, showscale=false), layout)
PlotlyJS.savefig(p_ave_pdf, joinpath(folder,"est_lpdf.png"))