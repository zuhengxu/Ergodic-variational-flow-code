using AdvancedHMC, ProgressMeter
import Random
using Random: GLOBAL_RNG, AbstractRNG
function HMC_adaptation(
    h::AdvancedHMC.Hamiltonian,
    κ::AdvancedHMC.HMCKernel,
    θ::T,
    adaptor::AdvancedHMC.AbstractAdaptor=NoAdaptation(),
    n_adapts::Int=min(div(n_samples, 10), 1_000);
    verbose::Bool=true,
    progress::Bool=false,
    (pm_next!)::Function=AdvancedHMC.pm_next!
) where {T<:AbstractVecOrMat{<:AbstractFloat}}
    # Prepare containers to store sampling results
    n_keep = n_adapts
    θs, stats = Vector{T}(undef, n_keep), Vector{NamedTuple}(undef, n_keep)
    # Initial sampling
    h, t = AdvancedHMC.sample_init(GLOBAL_RNG, h, θ)
    # Progress meter
    pm = progress ? ProgressMeter.Progress(n_adapts, desc="Sampling", barlen=31) : nothing
    time = @elapsed for i = 1:n_adapts
        # Make a transition
        t = AdvancedHMC.transition(GLOBAL_RNG, h, κ, t.z)
        # Adapt h and κ; what mutable is the adaptor
        tstat = AdvancedHMC.stat(t)
        h, κ, isadapted = AdvancedHMC.adapt!(h, κ, adaptor, i, n_adapts, t.z.θ, tstat.acceptance_rate)
        tstat = merge(tstat, (is_adapt=isadapted,))
        # Update progress meter
        if progress
            # Do include current iteration and mass matrix
            pm_next!(pm, (iterations=i, tstat..., mass_matrix=h.metric))
        # Report finish of adapation
        elseif verbose && isadapted && i == n_adapts
            @info "Finished $n_adapts adapation steps" adaptor κ.τ.integrator h.metric
        end
        # Store sample
        θs[i], stats[i] = t.z.θ, tstat
    end

   # Report end of sampling
    if verbose
        EBFMI_est = AdvancedHMC.EBFMI(map(s -> s.hamiltonian_energy, stats))
        average_acceptance_rate = mean(map(s -> s.acceptance_rate, stats))
        if θ isa AbstractVector
            n_chains = 1
        else
            n_chains = size(θ, 2)
            # Make sure that arrays are on CPU before printing.
            EBFMI_est = convert(Vector{eltype(EBFMI_est)}, EBFMI_est)
            average_acceptance_rate = convert(
                Vector{eltype(average_acceptance_rate)},
                average_acceptance_rate
            )
            EBFMI_est = "[" * join(EBFMI_est, ", ") * "]"
            average_acceptance_rate = "[" * join(average_acceptance_rate, ", ") * "]"
        end
        @info "Finished $n_adapts adaptation steps for $n_chains chains in $time (s)" h κ EBFMI_est average_acceptance_rate
    end
    return AdvancedHMC.getϵ(adaptor), AdvancedHMC.getM⁻¹(adaptor)
end

function HMC_get_adapt(θ0, δ, L, ∇L, Madapt; nleapfrog= 10, verbose = true)

    # choose Mass matrix
    d = size(θ0, 1)
    metric = DiagEuclideanMetric(d)
    # define  hamiltonian system 
    # ajoint is a user specified gradient system, returning a tuple (log_post, gradient) 
    ajoint = θ -> (L(θ), ∇L(θ))
    hamiltonian = Hamiltonian(metric, L, ajoint)
    # Define a leapfrog solver, with initial step size chosen heuristically
    init_ϵ = find_good_stepsize(hamiltonian, θ0)
    integrator = Leapfrog(init_ϵ)

    # combined adapatation scheme 
    proposal = AdvancedHMC.StaticTrajectory(integrator, nleapfrog)
    adaptor = NaiveHMCAdaptor(MassMatrixAdaptor(metric), StepSizeAdaptor(δ, integrator)) # combined adaptaiton scheme using stan window adaptaiton
    ϵ0, invM0 = AdvancedHMC.getϵ(adaptor), AdvancedHMC.getM⁻¹(adaptor)
    @info "[AdvancedHMC] initialization" ϵ0 invM0

    # Run the sampler to draw samples from the specified Gaussian, where
    #   - `samples` will store the samples
    #   - `stats` will store diagnostic statistics for each sample
    stepsize, invM = HMC_adaptation(hamiltonian, proposal, θ0, adaptor, Madapt; progress=verbose)

    @info "[AdavancedHMC] sampling complete"
    return stepsize, invM
end
# ##############
# # example 
# ################
# using Distributions, ForwardDiff
# D = 10
# L(θ) = logpdf(MvNormal(zeros(D), I), θ)
# ∇L(θ) = ForwardDiff.gradient(L, θ)
# ϵ, Minv = HMC_get_adapt(100*ones(D), 0.65, L, ∇L, 10000; nleapfrog = 10)



