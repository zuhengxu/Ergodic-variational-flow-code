using PDMats, Parameters, LinearAlgebra 

@with_kw struct LOL
    d::Int64
    N_steps::Int64
    # damp coef
    γ::Real = 1.0
    # stepsize and inverse_massmatrix
    ϵ::Real = 0.2 
    invMass::PDMats.AbstractPDMat = PDMat(I(d))
end

