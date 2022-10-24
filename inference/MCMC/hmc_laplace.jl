include("leapfrog.jl")
include("init_stepsize.jl")


##################3
# slight modificaiton to alg 5 of NUTS paper 
################3

function hmc_laplace(θ0, ϵ,  δ, n_lfrg, L, ∇L, M, Madapt; verbose = true)
    """
    - θ: init pos 
    - ϵ: setpsize
    - δ: target acc ratio   
    - L, ∇L: target 
    - M : nsamples
    - Madapt : num of samples used for adaptation
    """

    if verbose
        # progress bar
        prog_bar = ProgressMeter.Progress(M+Madapt, dt=0.5, barglyphs=ProgressMeter.BarGlyphs("[=> ]"), barlen=50, color=:yellow)
    end
    θs = Vector{typeof(θ0)}(undef, M + Madapt + 1)

    θs[1]= θ0 
    # # ϵ  =find_reasonable_ϵ(θ0, L, ∇L)
    μ, γ, t_0, κ = log(10) .+ log.(ϵ), 0.05, 10, 0.75
    logϵbar, Hbar = 0.0, 0.0

    if verbose
        @info "[HMC] start sampling for $M samples with inital ϵ = $ϵ"
    end

    for m in 2:M+1 + Madapt
        r = rand(Laplace(), size(θ0, 1))
        θs[m], θ1, r1 = θs[m-1], θs[m-1], r

        for i = 1:n_lfrg 
            θ1, r1 = leapfrog_laplace(θ1, r1, ϵ, ∇L)
        end 

        loga = min(0.0, L(θ1) - L(θs[m - 1]) - sum(abs, r1) + sum(abs, r))
        if log(rand()) < loga 
            θs[m] = θ1
        end

        if verbose
            # update progress bar
            ProgressMeter.next!(prog_bar)
        end

    end
    if verbose
        @info "[HMC] sampling complete with final adapt ϵ = $ϵ"
    end
    
    M = reduce(hcat, θs[Madapt+2:end])
    # if size(M) is a row matrix, reshape to make it a N×1 matrix
    return size(M, 1) > 1 ? Matrix(M') : reshape(M, size(M, 2), 1)
end