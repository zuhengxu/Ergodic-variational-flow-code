using JLD

function lpdf_neo_save(o::NEO.NEOobj, X, Y; res_dir = "result/", res_name = "lpdf_neo.jld")
    n1, n2 = size(X, 1), size(Y, 1)
    T = Matrix{Float64}(undef, n1, n2)
    prog_bar = ProgressMeter.Progress(n1*n2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    rv = MvNormalCanon(zeros(o.d), zeros(o.d), o.invMass)
    p = rand(rv)
    @threads for i = 1:n1
        for j=1:n2 
            T[i, j] = NEO.neo_lpdf(o_neo, [X[i], Y[j]], p)
            ProgressMeter.next!(prog_bar)
        end
    end
    JLD.save(joinpath(res_dir, res_name), "lpdf", T, "X", X, "Y", Y)
    return T 
end

function lpdf_est_save(o::HamFlow, a::HF_params, X, Y; 
                        n_mcmc, nB, error_checking = false, res_dir = "result/",res_name = "lpdf_est.jld")

    Ds, E = log_density_slice_2d(X, Y, o.ρ_sampler(2), rand(), o, a.leapfrog_stepsize, a.μ, a.D, ErgFlow.inv_refresh, n_mcmc; nBurn = nB)
    Dd = [logp([x, y]) for x in X , y in Y]
    JLD.save(joinpath(res_dir, res_name), "lpdf", Dd, "lpdf_est", Ds, "X", X, "Y", Y, 
            "ϵ", a.leapfrog_stepsize, "μ", a.μ, "D", a.D)
    return Ds, Dd, E
end

function log_density_slice_2d(X, Y, ρ, u, o::ErgodicFlow, ϵ, μ, D, inv_ref::Function, n_mcmc::Int; nBurn = 0, error_check = false) 
        n1, n2 = size(X, 1), size(Y, 1)
        T = Matrix{Float64}(undef, n1, n2)
        E = Matrix{Float64}(undef, n1, n2)
        prog_bar = ProgressMeter.Progress(n1*n2, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
        @threads for i = 1:n1
        # println("$i / $n1")
        for j=1:n2 
            # this step is bit hacky since we do not use u
            T[i, j], E[i, j] = ErgFlow.log_density_est([X[i],Y[j]], ρ, u, o, ϵ, μ, D, inv_ref, n_mcmc; nBurn = nBurn)
            # T[i, j] = logmeanexp(@view(A[nBurn+1:end])) 
            ProgressMeter.next!(prog_bar)
        end
    end
    return T, E 
end


function running_mean(T; dims=1) 
    n = size(T, dims)
    return cumsum(T, dims = dims)./[1:n ;]
end

function running_var(T; dims=1) 
    X2 = T.^2
    M = running_mean(T; dims = dims)
    Ex2 = running_mean(X2; dims = dims)
    return Ex2 .- M.^2
end

