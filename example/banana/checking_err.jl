include("model_2d.jl")
include("../common/plotting.jl")
include("../common/result.jl")
include("../common/error.jl")
import PlotlyJS as pjs
Random.seed!(1)
###########3
#  fwd/bwd flow err
###########
n_lfrg = 50
o_lap = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randl, ErgFlow.lpdf_laplace_std, ErgFlow.∇lpdf_laplace_std, ErgFlow.cdf_laplace_std, ErgFlow.invcdf_laplace_std, ErgFlow.pdf_laplace_std,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

o_norm = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_normal, ErgFlow.invcdf_normal, ErgFlow.pdf_normal,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

o_log = HamFlow(d, n_lfrg, logp, ∇logp, randn, logq, 
        ErgFlow.randlogistic, ErgFlow.lpdf_logistic, ErgFlow.∇lpdf_logistic, ErgFlow.cdf_logistic, ErgFlow.invcdf_logistic, ErgFlow.pdf_logistic,  
        ErgFlow.constant, ErgFlow.mixer, ErgFlow.inv_mixer)

rot_mat = ErgFlow.rotation_mat(1.0)
rot_mat_inv = ErgFlow.rotation_mat(-1.0)

o = HamFlowRot(d, n_lfrg, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_exp, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_exp, ErgFlow.invcdf_exp, ErgFlow.pdf_normal,  
        ErgFlow.constant, rot_mat, rot_mat_inv, ErgFlow.mixer, ErgFlow.inv_mixer)
a = ErgFlow.HF_params(0.01*ones(d), μ, D)

#############
# computes fwd/bwd error for different momentum dist or refresh scheme
#############
stability_plot(o_lap, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_lap.jld")
stability_plot(o_norm, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_norm.jld")
stability_plot(o_log, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, res_name = "stab_log.jld")
stability_plot(o, a, [10, 50, 100, 200, 500, 1000, 1500, 2000, 3000]; nsample = 100, refresh = ErgFlow.pseudo_refresh, inv_ref = ErgFlow.inv_refresh, res_name = "stab_norm_new.jld")


###########
#  fwd and bwd trajectory
##########
# # save samples every 10 leapfrogs 
# T_lfrg, M_lfrg, T, M, _ = ErgFlow.flow_fwd_save(o,a.leapfrog_stepsize, pseudo_refresh, z0, randn(2), rand(), 3000; freq = 10)
# plot(M_lfrg[5:5:end, :])

Random.seed!(1)
z0 = randn(2) .* a.D + a.μ
ρ0 = ErgFlow.randn(2)
# ρ0 = ErgFlow.randl(2)
u0 = rand()

x = -20:0.1:20
y = -15:0.1:30
xm = -8:0.1:8
ym = -8:0.1:8
o = HamFlowRot(d, 50, logp, ∇logp, randn, logq, 
        randn, ErgFlow.lpdf_exp, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.cdf_exp, ErgFlow.invcdf_exp, ErgFlow.pdf_normal,  
        ErgFlow.stream, rot_mat, rot_mat_inv, ErgFlow.mixer, ErgFlow.inv_mixer)
a = ErgFlow.HF_params(0.01*ones(d), μ, D)
T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd = flow_trace_plot(o, a, x, y, xm, ym, z0, ρ0, u0; n_mcmc = 100,fig_dir = "figure/", name = "flow_trace.png")

Random.seed!(1)
n_lfrg = 500
n_mcmc = 100
r_state = ErgFlow.warm_start(∇logp, ErgFlow.∇lpdf_normal, a, d, 5000, n_mcmc, n_lfrg)
o_sb = SB_refresh(d, n_lfrg, logp, ∇logp, randn, logq, randn, ErgFlow.lpdf_normal, ErgFlow.∇lpdf_normal, ErgFlow.pdf_normal, r_state)
T_fwd, M_fwd, U_fwd, T_bwd, M_bwd, U_bwd = flow_trace_plot(o_sb, a, x, y, xm, ym, z0, ρ0, u0; refresh = ErgFlow.refresh_sb, inv_ref = ErgFlow.inv_refresh_sb,  n_mcmc = n_mcmc,fig_dir = "figure/", name = "flow_trace.png")



#################
# see error of each transformation for a batch of samples 
################3
n_mcmc = 100
leap_err, ref_err, total_err, E = ErgFlow.flow_fwd_err_tr(o,a.leapfrog_stepsize, pseudo_refresh, inv_refresh, z0, ρ0, u0, n_mcmc)
leap_err, ref_err, total_err, E = ErgFlow.flow_fwd_err_tr_sb(o_sb,a.leapfrog_stepsize, z0, ρ0, u0)
p1 = plot(leap_err, alpha = 0.4, label = "lfrg")
p2 = plot(ref_err, alpha = 0.4, label = "refresh")
p3 = plot(total_err, alpha = 0.4, label = "1step")
p_err = plot(p1, p2, p3, layout = (3, 1), title = "err = $E, #ref = $n_mcmc")
savefig(p_err, "figure/single_sample_err.png")

# l_e, r_e, t_e, E = ErgFlow.flow_fwd_err_tr(o_lap, a, pseudo_refresh, inv_refresh, n_mcmc)
# lay = @layout [grid(3, 1) a{0.4w}]
# p1 = plot(1:n_mcmc, vec(median(l_e, dims = 2)), ribbon = get_percentiles(l_e), lw = 1, label = "lfrg")
# p2 = plot(1:n_mcmc, vec(median(r_e, dims = 2)), ribbon = get_percentiles(r_e), lw = 1, label = "ref")
# p3 = plot(1:n_mcmc, vec(median(t_e, dims = 2)), ribbon = get_percentiles(t_e), lw = 1, label = "1step")
# p4 = histogram(E, bins = 100, label="fwd err")
# p_err = plot(p1, p2, p3, p4, layout = lay) 
# savefig(p_err, "figure/batch_sample_err.png")

# ##############33
# # see samples
# ################

# Random.seed!(1)
# x = -20:0.1:20
# y = -15:0.1:30
# scatter_gif(o, a, x, y; momentum = "norm", freq = 25,
#         n_sample = 100, n_mcmc = 10, bins = 50, name= "trace.gif")

# scatter_gif(o_norm, a, x, y; momentum = "norm", refresh = pseudo_refresh, freq = 25,
#         n_sample = 100, n_mcmc = 10, bins = 50, name= "trace_norm.gif")

# scatter_gif(o_log, a, x, y; momentum = "logistic", refresh = pseudo_refresh, freq = 25,
#         n_sample = 100, n_mcmc = 10, bins = 50, name= "trace_log.gif")

# scatter_gif(o_lap, a, x, y; momentum = "laplace", refresh = pseudo_refresh, freq = 25,
#         n_sample = 100, n_mcmc = 10, bins = 50, name= "trace_lap.gif")

# scatter_plot(o_lap, x, y; μ=μ, D=D, ϵ = 0.02*ones(d), n_sample = 100, n_mcmc = 500, nB = 0, bins = 50, name= "lap_scatter.png")
