using Distributions, LinearAlgebra, Random, StatsBase
using ForwardDiff
using ErgFlow
using Base.Threads:@threads
using JLD, JLD2, DataFrames, CSV 
using Zygote:Buffer, ignore, gradient, @ignore, @adjoint
include("../../inference/SVI/svi.jl")


function data_load(dat)
```
load and process .npz data
```
    Dat = CSV.read(dat, DataFrame; header = 1)
    # # turn String31 into Float64
    # for c in names(Dat)
    #     if eltype(Dat[!,c]) != Float64
    #         Dat[!, c] = parse.(Float64, Dat[!, c])
    #     end
    # end
    Zr, Y = Dat[:, 1:end-1], Dat[:,end]
    Zr[:, 3] .= 140 .- Zr[:,3]
    Zr = log.(Zr)
    N, p = size(Zr)
    Zr = Matrix(Zr)
    # standarize dataset 
    Zr = (Zr .- mean(Zr, dims =1)) ./ std(Zr, dims = 1)
    Y = (Y .- mean(Y))./std(Y)
    # add intercept
    Z = hcat(ones(N), Zr)
    # turn down datasize to make2 post looks wierd
    # Random.seed!(1)
    # idx = sample(1:N, 100, replace = false)
    return Matrix(Z), Y, N, p +1
end

##################
# load and process dataset
#################3
cd("/arc/project/st-tdjc-1/mixflow/Ergodic-variational-flow-code/example/heavy_reg")
X, Y, N, d = data_load("data/creatinine.csv")


#######################
# models ( t₅(Xβ, 1)-distirbution lin reg with heavy tailed prior, β ∼ Cauchy(0, 1))
#######################
function log_pr_β(β)
    return -d*log(π) - sum(log1p, β.^2.0)
end

#  β:= β₁,...,βₚ,β_{p+1}
function log_lik(β)
    diffs = Y .- X * β
    return -3.0*sum(log, 5.0 .+ diffs.^2.0) 
end

function logp(β)
    log_lik(β) + log_pr_β(β)
end


@inline function ∇log_pr_β(β::Real)
    return -2.0*β/(1.0 + β^2.0)
end
function ∇log_lik(β)
    diffs = Y .- X * β
    A = @. 6.0*diffs/(5.0 + diffs^2)
    return X'*A
end
∇logp(β) = ∇log_lik(β) .+ ∇log_pr_β.(β) 


# Customize zygote gardient
@adjoint logp(z) = logp(z), Δ -> (Δ*∇logp(z), )

########################
# SVI
########################
logq(x, μ, D) =  -0.5*d*log(2π) - sum(log, abs.(D)) - 0.5*sum(abs2, (x.-μ)./(D .+ 1e-8))
∇logq(x, μ, D) = (μ .- x)./(D .+ 1e-8)


if ! isdir("figure")
    mkdir("figure")
end 
if ! isdir("result")
    mkdir("result")
end 