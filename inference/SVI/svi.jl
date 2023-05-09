module SVI


    using ForwardDiff, Flux, LinearAlgebra, Distributions, Random, StatsBase, SpecialFunctions

    abstract type StochasticVI end
    abstract type params end

    ##################################3
    # mean field Gaussian variational inference
    ####################################
     include("MFvi.jl")

end