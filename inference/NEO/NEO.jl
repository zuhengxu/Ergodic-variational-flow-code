module NEO
# a minimal implementation of NEO-IS and NEO-MCMC: https://openreview.net/pdf?id=76tTYokjtG
# some code refer to "https://github.com/Achillethin/NEO_non_equilibrium_sampling"
# Basic setting: 
# 1. w_k = 1_[N-1] (only involving N-1 forward transitions)
# 2. using same reference distribution as ErgFlow  
# 3. symplectic Euler (instead of leapfrog) integrator as described in paper
# 4. integration stepsize ϵ, invMass matrix M^{-1} using automatic HMC adaptation 


using LinearAlgebra, Distributions, Random, StatsBase, ProgressMeter, Parameters
using SpecialFunctions, LogExpFunctions
using Base.Threads: @threads




include("util.jl")
include("neoIS.jl")
include("neoMCMC.jl")

end